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
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.client.SessionConfig
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
    val languages: Seq[(Language, Vocabulary)],
    val dataConfig: DataConfig,
    val config: Model.Config,
    val optConfig: Model.OptConfig,
    val logConfig: Model.LogConfig = Model.LogConfig(),
    val evalDatasets: Seq[(String, FileParallelDataset)] = Seq.empty
) {
  protected val languageIds: Map[Language, Int] = languages.map(_._1).zipWithIndex.toMap

  protected val parameterManager: ParameterManager = config.parameterManager

  /** Each input consists of a tuple containing:
    *   - The source language ID.
    *   - The target language ID.
    *   - A tensor containing a padded batch of sentences consisting of word IDs, in the source language.
    *   - A tensor containing the sentence lengths for the aforementioned padded batch.
    */
  protected val input      = Input((INT32, INT32, STRING, INT32), (Shape(), Shape(), Shape(-1, -1), Shape(-1)))
  protected val trainInput = Input((INT32, STRING, INT32), (Shape(), Shape(-1, -1), Shape(-1)))

  protected val estimator: tf.learn.Estimator[
      TFBatchWithLanguagesT, TFBatchWithLanguages, TFBatchWithLanguagesD, TFBatchWithLanguagesS, TFBatchWithLanguage,
      (TFBatchWithLanguagesT, TFBatchWithLanguageT), (TFBatchWithLanguages, TFBatchWithLanguage),
      (TFBatchWithLanguagesD, TFBatchWithLanguageD), (TFBatchWithLanguagesS, TFBatchWithLanguageS),
      (TFBatchWithLanguage, TFBatch)] = tf.createWithNameScope(name) {
    val model = learn.Model.supervised(
      input, inferLayer, trainLayer, trainInput, trainInputLayer, lossLayer, optConfig.optimizer,
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
      var datasets = Seq.empty[(String, FileParallelDataset)]
      for (datasetType <- Seq(Train, Dev, Test))
        datasets ++= evalDatasets.map(d => (s"${d._1}/$datasetType", d._2.filterTypes(datasetType)))
      datasets = datasets.filter(_._2.nonEmpty)
      if (datasets.nonEmpty) {
        hooks += tf.learn.Evaluator(
          log = true, summariesDir, Inputs.createEvalDatasets(dataConfig, config, datasets, languages), Seq(BLEU()),
          StepHookTrigger(logConfig.logEvalSteps), triggerAtEnd = true, name = "Evaluation")
      }
    }

    // Create estimator.
    tf.learn.InMemoryEstimator(
      model, tf.learn.Configuration(
        workingDir = Some(config.env.workingDir),
        sessionConfig = Some(SessionConfig(allowSoftPlacement = Some(true))),
        randomSeed = config.env.randomSeed),
      trainHooks = hooks)
  }

  def train(datasets: Seq[FileParallelDataset], stopCriteria: StopCriteria): Unit = {
    estimator.train(Inputs.createTrainDataset(
      dataConfig, config, datasets, languages, repeat = true, isEval = false), stopCriteria)
  }

  def train(dataset: FileParallelDataset, stopCriteria: StopCriteria): Unit = {
    train(Seq(dataset), stopCriteria)
  }

  def translate(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataset: FileParallelDataset
  ): Iterator[(TFBatchWithLanguagesT, TFBatchWithLanguageT)] = {
    estimator.infer(Inputs.createInputDataset(dataConfig, config, dataset, srcLanguage, tgtLanguage, languages))
  }

//  def translate(
//      srcLanguage: (Language, Vocabulary),
//      tgtLanguage: (Language, Vocabulary),
//      input: (Tensor, Tensor)
//  ): Iterator[((Tensor, Tensor, Tensor, Tensor), (Tensor, Tensor, Tensor))] = {
//    translate(srcLanguage._1, tgtLanguage._1, TensorParallelDataset(
//      name = "TranslateTemp", vocabularies = Map(srcLanguage, tgtLanguage),
//      tensors = Map(srcLanguage._1 -> Seq(input))))
//  }
//
//  def evaluate(
//      datasets: Seq[(String, ParallelDataset)],
//      metrics: Seq[MTMetric],
//      maxSteps: Long = -1L,
//      saveSummaries: Boolean = true,
//      name: String = null
//  ): Seq[Tensor] = {
//    estimator.evaluate(Model.createEvalDatasets(datasets, languageIds), metrics, maxSteps, saveSummaries, name)
//  }

  protected def trainInputLayer: Layer[TFBatchWithLanguage, TFBatch] = {
    new Layer[TFBatchWithLanguage, TFBatch](name) {
      override val layerType: String = "InputTrain"

      override protected def _forward(
          input: (Output, Output, Output),
          mode: Mode
      ): (Output, Output) = {
        parameterManager.initialize(languages)
        tf.createWithNameScope("TrainInputsToWordIDs") {
          (mapToWordIds(input._1, input._2), input._3)
        }
      }
    }
  }

  protected def trainLayer: Layer[(TFBatchWithLanguages, TFBatchWithLanguage), TFBatchWithLanguage] = {
    new Layer[(TFBatchWithLanguages, TFBatchWithLanguage), TFBatchWithLanguage](name) {
      override val layerType: String = "TrainLayer"

      override protected def _forward(
          input: (TFBatchWithLanguages, TFBatchWithLanguage),
          mode: Mode
      ): TFBatchWithLanguage = {
        parameterManager.initialize(languages)
        parameterManager.setContext((input._1._1, input._1._2))
        val srcSequence = mapToWordIds(input._1._1, input._1._3)
        val tgtSequence = mapToWordIds(input._2._1, input._2._2)
        val srcMapped = (input._1._1, input._1._2, srcSequence, input._1._4)
        val tgtMapped = (tgtSequence, input._2._3)
        val state = tf.createWithVariableScope("Encoder")(encoder(srcMapped, mode))
        val output = tf.createWithVariableScope("Decoder")(decoder(srcMapped, Some(tgtMapped), Some(state), mode))
        (input._1._2, output._1, output._2)
      }
    }
  }

  protected def inferLayer: Layer[TFBatchWithLanguages, TFBatchWithLanguage] = {
    new Layer[TFBatchWithLanguages, TFBatchWithLanguage](name) {
      override val layerType: String = "InferLayer"

      override protected def _forward(input: TFBatchWithLanguages, mode: Mode): TFBatchWithLanguage = {
        parameterManager.initialize(languages)
        parameterManager.setContext((input._1, input._2))
        val srcSequence = mapToWordIds(input._1, input._3)
        val srcMapped = (input._1, input._2, srcSequence, input._4)
        val state = tf.createWithVariableScope("Encoder")(encoder(srcMapped, mode))
        val output = tf.createWithVariableScope("Decoder")(decoder(srcMapped, None, Some(state), mode))
        (input._2, output._1, output._2)
      }
    }
  }

  protected def lossLayer: Layer[(TFBatchWithLanguage, TFBatch), Output] = {
    new Layer[(TFBatchWithLanguage, TFBatch), Output](name) {
      override val layerType: String = "Loss"

      override protected def _forward(
          input: (TFBatchWithLanguage, TFBatch),
          mode: Mode
      ): Output = tf.createWithNameScope("Loss") {
        // TODO: Handle this shift more efficiently.
        // Shift the target sequence one step backward so the decoder is evaluated based using the correct previous
        // word used as input, rather than the previous predicted word.
        val tgtEosId = parameterManager
            .lookupTable(input._1._1)(tf.constant(dataConfig.endOfSequenceToken))
            .cast(INT32)
        val tgtSequence = tf.concatenate(Seq(
          input._2._1, tf.fill(INT32, tf.stack(Seq(tf.shape(input._2._1)(0), 1)))(tgtEosId)), axis = 1)
        val tgtSequenceLength = input._2._2 + 1
        val lossValue = loss(input._1._2, tgtSequence, tgtSequenceLength)
        tf.summary.scalar("Loss", lossValue)
        lossValue
      }
    }
  }

  protected def mapToWordIds(language: Output, wordSequence: Output): Output = {
    parameterManager.lookupTable(language)(wordSequence).cast(INT32)
  }

  /**
    *
    * @param  input        Tuple containing two tensors:
    *                        - `INT32` tensor with shape `[batchSize, inputLength]`, containing the sentence word IDs.
    *                        - `INT32` tensor with shape `[batchSize]`, containing the sequence lengths.
    * @param  mode    Current learning mode (e.g., training or evaluation).
    * @return   Tuple containing two tensors:
    *           - Encoder output, with shape `[batchSize, inputLength, hiddenSize]`.
    *           - Encoder-decoder attention bias and mask weights, with shape `[batchSize, inputLength]`.
    */
  protected def encoder(input: TFBatchWithLanguages, mode: Mode): S

  /**
    *
    * @return Tensor with shape `[batchSize, length, 1, hiddenSize]`.
    */
  protected def decoder(
      encoderInput: TFBatchWithLanguages,
      input: Option[TFBatch],
      state: Option[S],
      mode: Mode
  ): TFBatch

  protected def loss(predictedSequences: Output, targetSequences: Output, targetSequenceLengths: Output): Output = {
    val (lossSum, _) = Common.paddedCrossEntropy(
      predictedSequences, targetSequences, targetSequenceLengths, config.labelSmoothing, timeMajor = config.timeMajor)
    lossSum / tf.size(targetSequenceLengths).cast(FLOAT32)
  }
}

object Model {
  class Config protected (
      val env: Environment,
      val parameterManager: ParameterManager,
      val labelSmoothing: Float,
      val timeMajor: Boolean,
      val summarySteps: Int,
      val checkpointSteps: Int)

  object Config {
    def apply(
        env: Environment,
        parameterManager: ParameterManager,
        labelSmoothing: Float = 0.0f,
        timeMajor: Boolean = false,
        summarySteps: Int = 100,
        checkpointSteps: Int = 1000
    ): Config = {
      new Config(env, parameterManager, labelSmoothing, timeMajor, summarySteps, checkpointSteps)
    }
  }

  class OptConfig protected (
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

  class LogConfig protected (
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
}
