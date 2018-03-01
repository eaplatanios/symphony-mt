/* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package org.platanios.symphony.mt.models

import org.platanios.symphony.mt.{Environment, Language}
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.evaluation.BLEU
import org.platanios.symphony.mt.models.helpers.Common
import org.platanios.symphony.mt.models.hooks.TrainingLogger
import org.platanios.symphony.mt.vocabulary.{Vocabularies, Vocabulary}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.{Mode, StopCriteria}
import org.platanios.tensorflow.api.learn.layers.{Input, Layer}
import org.platanios.tensorflow.api.learn.hooks.StepHookTrigger
import org.platanios.tensorflow.api.ops.training.optimizers.{GradientDescent, Optimizer}

// TODO: Move embeddings initializer to the configuration.
// TODO: Add support for optimizer schedules (e.g., Adam for first 1000 steps and then SGD with a different learning rate.
// TODO: Customize evaluation metrics, hooks, etc.

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Model[S] protected (
    val name: String,
    val languages: Map[Language, Vocabulary],
    val dataConfig: DataConfig,
    val config: Model.Config,
    val optConfig: Model.OptConfig,
    val logConfig: Model.LogConfig = Model.LogConfig(),
    val evalDatasets: Seq[(String, ParallelDataset)] = Seq.empty
) {
  protected val languageIds: Map[Language, Int] = languages.keys.zipWithIndex.toMap

  protected val parametersManager: ParametersManager[Seq[Language], (Output, Output)] = config.parametersManager

  /** Each input consists of a tuple containing:
    *   - The source language ID.
    *   - The target language ID.
    *   - A tensor containing a padded batch of sentences consisting of word IDs, in the source language.
    *   - A tensor containing the sentence lengths for the aforementioned padded batch.
    */
  protected val input      = Input((INT32, INT32, INT32, INT32), (Shape(), Shape(), Shape(-1, -1), Shape(-1)))
  protected val trainInput = Input((INT32, INT32), (Shape(-1, -1), Shape(-1)))

  protected val estimator: tf.learn.Estimator[
      (Tensor, Tensor, Tensor, Tensor), (Output, Output, Output, Output),
      (DataType, DataType, DataType, DataType), (Shape, Shape, Shape, Shape), (Output, Output, Output),
      ((Tensor, Tensor, Tensor, Tensor), (Tensor, Tensor)), ((Output, Output, Output, Output), (Output, Output)),
      ((DataType, DataType, DataType, DataType), (DataType, DataType)), ((Shape, Shape, Shape, Shape), (Shape, Shape)),
      ((Output, Output, Output), (Output, Output))] = tf.createWithNameScope(name) {
    val model = learn.Model.supervised(
      input, inferLayer, trainLayer, trainInput, lossLayer, optConfig.optimizer,
      tf.learn.ClipGradientsByGlobalNorm(optConfig.maxGradNorm), optConfig.colocateGradientsWithOps)
    val summariesDir = config.env.workingDir.resolve("summaries")

    // Create estimator hooks.
    var hooks = Set[tf.learn.Hook](
      // tf.learn.LossLogger(trigger = tf.learn.StepHookTrigger(1)),
      tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = StepHookTrigger(100)),
      tf.learn.SummarySaver(summariesDir, StepHookTrigger(config.summarySteps)),
      tf.learn.CheckpointSaver(config.env.workingDir, StepHookTrigger(config.checkpointSteps)))

    // Add logging hooks.
    if (logConfig.logLossSteps > 0)
      hooks += TrainingLogger(log = true, trigger = StepHookTrigger(logConfig.logLossSteps))
    if (logConfig.logEvalSteps > 0) {
      var datasets = Seq.empty[(String, ParallelDataset)]
      for (datasetType <- Seq(Train, Dev, Test))
        datasets ++= evalDatasets.map(d => (s"${d._1}/$datasetType", d._2.filterTypes(datasetType)))
      datasets = datasets.filter(_._2.nonEmpty)
      if (datasets.nonEmpty) {
        val batchSize = dataConfig.evaluateBatchSize
        val prefetchBufferSize = if (dataConfig.bufferSize == -1L) 1024L * batchSize else dataConfig.bufferSize
        hooks += tf.learn.Evaluator(
          log = true, summariesDir, Model.createEvalDatasets(
            datasets, languageIds, prefetchBufferSize = prefetchBufferSize), Seq(BLEU()),
          StepHookTrigger(logConfig.logEvalSteps), triggerAtEnd = true, name = "Evaluation")
      }
    }

    // Create estimator.
    tf.learn.InMemoryEstimator(
      model, tf.learn.Configuration(Some(config.env.workingDir), randomSeed = config.env.randomSeed),
      trainHooks = hooks)
  }

  def train(datasets: Seq[ParallelDataset], stopCriteria: StopCriteria): Unit = {
    val batchSize = dataConfig.trainBatchSize
    val prefetchBufferSize = if (dataConfig.bufferSize == -1L) 1024L * batchSize else dataConfig.bufferSize
    estimator.train(Model.createTrainDataset(
      datasets, languageIds, repeat = true, isEval = false, prefetchBufferSize = prefetchBufferSize), stopCriteria)
  }

  def train(dataset: ParallelDataset, stopCriteria: StopCriteria): Unit = {
    train(Seq(dataset), stopCriteria)
  }

  def translate(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataset: ParallelDataset
  ): Iterator[((Tensor, Tensor, Tensor, Tensor), (Tensor, Tensor, Tensor))] = {
    estimator.infer(Model.createInputDataset(srcLanguage, tgtLanguage, dataset, languageIds))
  }

  def translate(
      srcLanguage: (Language, Vocabulary),
      tgtLanguage: (Language, Vocabulary),
      input: (Tensor, Tensor)
  ): Iterator[((Tensor, Tensor, Tensor, Tensor), (Tensor, Tensor, Tensor))] = {
    translate(srcLanguage._1, tgtLanguage._1, TensorParallelDataset(
      name = "TranslateTemp", vocabularies = Map(srcLanguage, tgtLanguage),
      tensors = Map(srcLanguage._1 -> Seq(input))))
  }

//  def evaluate(
//      datasets: Seq[(String, ParallelDataset)],
//      metrics: Seq[MTMetric],
//      maxSteps: Long = -1L,
//      saveSummaries: Boolean = true,
//      name: String = null
//  ): Seq[Tensor] = {
//    estimator.evaluate(Model.createEvalDatasets(datasets, languageIds), metrics, maxSteps, saveSummaries, name)
//  }

  protected def trainLayer: Layer[((Output, Output, Output, Output), (Output, Output)), (Output, Output, Output)] = {
    new Layer[((Output, Output, Output, Output), (Output, Output)), (Output, Output, Output)](name) {
      override val layerType: String = "TrainLayer"

      override protected def _forward(
          input: ((Output, Output, Output, Output), (Output, Output)),
          mode: Mode
      ): (Output, Output, Output) = {
        parametersManager.initialize(Some(languages.keys.toSeq))
        parametersManager.setContext((input._1._1, input._1._2))
        val vocabularies = Vocabularies(languages, config.embeddingsSize)
        val state = tf.createWithVariableScope("Encoder")(encoder(input._1, vocabularies, mode))
        val output = tf.createWithVariableScope("Decoder") {
          decoder(input._1, vocabularies, Some(input._2), Some(state), mode)
        }
        (input._1._2, output._1, output._2)
      }
    }
  }

  protected def inferLayer: Layer[(Output, Output, Output, Output), (Output, Output, Output)] = {
    new Layer[(Output, Output, Output, Output), (Output, Output, Output)](name) {
      override val layerType: String = "InferLayer"

      override protected def _forward(input: (Output, Output, Output, Output), mode: Mode): (Output, Output, Output) = {
        parametersManager.initialize(Some(languages.keys.toSeq))
        parametersManager.setContext((input._1, input._2))
        val vocabularies = Vocabularies(languages, config.embeddingsSize)
        val state = tf.createWithVariableScope("Encoder")(encoder(input, vocabularies, mode))
        val output = tf.createWithVariableScope("Decoder")(decoder(input, vocabularies, None, Some(state), mode))
        (input._2, output._1, output._2)
      }
    }
  }

  protected def lossLayer: Layer[((Output, Output, Output), (Output, Output)), Output] = {
    new Layer[((Output, Output, Output), (Output, Output)), Output](name) {
      override val layerType: String = "Loss"

      override protected def _forward(
          input: ((Output, Output, Output), (Output, Output)),
          mode: Mode
      ): Output = tf.createWithNameScope("Loss") {
        // TODO: Recreating the vocabularies here may be inefficient.
        val vocabularies = Vocabularies(languages, config.embeddingsSize)
        // TODO: Handle this shift more efficiently.
        // Shift the target sequence one step backward so the decoder is evaluated based using the correct previous
        // word used as input, rather than the previous predicted word.
        val tgtEosId = vocabularies.lookupTable(input._1._1)(tf.constant(dataConfig.endOfSequenceToken)).cast(INT32)
        val tgtSequence = tf.concatenate(Seq(
          input._2._1,
          tf.fill(INT32, tf.stack(Seq(tf.shape(input._2._1)(0), 1)))(tgtEosId)), axis = 1)
        val tgtSequenceLength = input._2._2 + 1
        val lossValue = loss(input._1._2, tgtSequence, tgtSequenceLength)
        tf.summary.scalar("Loss", lossValue)
        lossValue
      }
    }
  }

  /**
    *
    * @param  input        Tuple containing two tensors:
    *                        - `INT32` tensor with shape `[batchSize, inputLength]`, containing the sentence word IDs.
    *                        - `INT32` tensor with shape `[batchSize]`, containing the sequence lengths.
    * @param  vocabularies
    * @param  mode    Current learning mode (e.g., training or evaluation).
    * @return   Tuple containing two tensors:
    *           - Encoder output, with shape `[batchSize, inputLength, hiddenSize]`.
    *           - Encoder-decoder attention bias and mask weights, with shape `[batchSize, inputLength]`.
    */
  protected def encoder(
      input: (Output, Output, Output, Output),
      vocabularies: Vocabularies,
      mode: Mode
  ): S

  /**
    *
    * @return Tensor with shape `[batchSize, length, 1, hiddenSize]`.
    */
  protected def decoder(
      encoderInput: (Output, Output, Output, Output),
      vocabularies: Vocabularies,
      input: Option[(Output, Output)],
      state: Option[S],
      mode: Mode
  ): (Output, Output)

  protected def loss(predictedSequences: Output, targetSequences: Output, targetSequenceLengths: Output): Output = {
    val (lossSum, _) = Common.paddedCrossEntropy(
      predictedSequences, targetSequences, targetSequenceLengths, config.labelSmoothing, timeMajor = config.timeMajor)
    lossSum / tf.size(targetSequenceLengths).cast(FLOAT32)
  }
}

object Model {
  class Config protected(
      val env: Environment,
      val embeddingsSize: Int,
      val parametersManager: ParametersManager[Seq[Language], (Output, Output)],
      val labelSmoothing: Float,
      val timeMajor: Boolean,
      val summarySteps: Int,
      val checkpointSteps: Int)

  object Config {
    def apply(
        env: Environment,
        embeddingsSize: Int,
        parametersManager: ParametersManager[Seq[Language], (Output, Output)] = DefaultParametersManager(
          tf.VarianceScalingInitializer(
            1.0f,
            tf.VarianceScalingInitializer.FanAverageScalingMode,
            tf.VarianceScalingInitializer.UniformDistribution)),
        labelSmoothing: Float = 0.0f,
        timeMajor: Boolean = false,
        summarySteps: Int = 100,
        checkpointSteps: Int = 1000
    ): Config = {
      new Config(env, embeddingsSize, parametersManager, labelSmoothing, timeMajor, summarySteps, checkpointSteps)
    }
  }

  class OptConfig protected(
      val maxGradNorm: Float,
      val optimizer: Optimizer,
      val colocateGradientsWithOps: Boolean)

  object OptConfig {
    def apply(
        maxGradNorm: Float = 5.0f,
        optimizer: Optimizer = GradientDescent(1.0f, learningRateSummaryTag = "LearningRate"),
        colocateGradientsWithOps: Boolean = true
    ): OptConfig = {
      new OptConfig(maxGradNorm, optimizer, colocateGradientsWithOps)
    }
  }

  class LogConfig protected(
      val logLossSteps: Int,
      val logEvalBatchSize: Int,
      val logEvalSteps: Int)

  object LogConfig {
    def apply(
        logLossSteps: Int = 100,
        logEvalBatchSize: Int = 512,
        logEvalSteps: Int = 1000
    ): LogConfig = {
      new LogConfig(logLossSteps, logEvalBatchSize, logEvalSteps)
    }
  }

  type TFInputDataset = tf.data.Dataset[
      (Tensor, Tensor, Tensor, Tensor), (Output, Output, Output, Output),
      (DataType, DataType, DataType, DataType), (Shape, Shape, Shape, Shape)]

  type TFTrainDataset = tf.data.Dataset[
      ((Tensor, Tensor, Tensor, Tensor), (Tensor, Tensor)),
      ((Output, Output, Output, Output), (Output, Output)),
      ((DataType, DataType, DataType, DataType), (DataType, DataType)),
      ((Shape, Shape, Shape, Shape), (Shape, Shape))]

  private[Model] def createInputDataset(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataset: ParallelDataset,
      languageIds: Map[Language, Int]
  ): () => TFInputDataset = () => {
    dataset.toTFMonolingual(srcLanguage)
        .map(
          d => (tf.constant(languageIds(srcLanguage)), tf.constant(languageIds(tgtLanguage)), d._1, d._2),
          name = s"AddInputLanguageIDs$srcLanguage$tgtLanguage")
        .asInstanceOf[TFInputDataset]
  }

  private[Model] def createTrainDataset(
      datasets: Seq[ParallelDataset],
      languageIds: Map[Language, Int],
      repeat: Boolean = true,
      isEval: Boolean = false,
      prefetchBufferSize: Long = 1024L
  ): () => TFTrainDataset = () => {
    tf.createWith(device = null) {
      val processedDatasets: Seq[TFTrainDataset] = datasets
          .map(_.filterLanguages(languageIds.keys.toSeq: _*))
          .flatMap(d => d.languagePairs().map(_ -> d))
          .map {
            case ((srcLanguage, tgtLanguage), dataset) =>
              dataset.toTFBilingual(srcLanguage, tgtLanguage, repeat = repeat, isEval = isEval)
                  .map(
                    d => ((
                        tf.constant(languageIds(srcLanguage)),
                        tf.constant(languageIds(tgtLanguage)),
                        d._1._1, d._1._2), d._2),
                    name = s"AddTrainLanguageIDs$srcLanguage$tgtLanguage")
                  .asInstanceOf[TFTrainDataset]
          }
      processedDatasets.reduce((d1, d2) => d1.concatenate(d2)).prefetch(prefetchBufferSize)
    }
  }

  private[Model] def createEvalDatasets(
      datasets: Seq[(String, ParallelDataset)],
      languageIds: Map[Language, Int],
      prefetchBufferSize: Long = 1024L
  ): Seq[(String, () => TFTrainDataset)] = {
    tf.createWith(device = null) {
      datasets
          .map(d => (d._1, d._2.filterLanguages(languageIds.keys.toSeq: _*)))
          .flatMap(d => d._2.languagePairs().map(l => (d._1, l) -> d._2))
          .map {
            case ((name, (srcLanguage, tgtLanguage)), dataset) =>
              (s"$name/${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}",
                  () => {
                    dataset.toTFBilingual(srcLanguage, tgtLanguage, repeat = false, isEval = true)
                        .map(
                          d => ((
                              tf.constant(languageIds(srcLanguage)),
                              tf.constant(languageIds(tgtLanguage)),
                              d._1._1, d._1._2), d._2),
                          name = s"AddTrainLanguageIDs$srcLanguage$tgtLanguage")
                        .asInstanceOf[TFTrainDataset].prefetch(prefetchBufferSize)
                  })
          }
    }
  }
}
