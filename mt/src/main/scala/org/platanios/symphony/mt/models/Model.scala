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
import org.platanios.symphony.mt.evaluation.{BLEU, MTMetric}
import org.platanios.symphony.mt.models.helpers.Common
import org.platanios.symphony.mt.models.hooks.TrainingLogger
import org.platanios.symphony.mt.vocabulary.Vocabulary
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
    val srcLanguage: Language,
    val srcVocabulary: Vocabulary,
    val tgtLanguage: Language,
    val tgtVocabulary: Vocabulary,
    val dataConfig: DataConfig,
    val config: Model.Config,
    val optConfig: Model.OptConfig,
    val logConfig: Model.LogConfig = Model.LogConfig(),
    val trainEvalDataset: () => TFBilingualDataset = null,
    val devEvalDataset: () => TFBilingualDataset = null,
    val testEvalDataset: () => TFBilingualDataset = null
) {
  /** Each input consists of a tuple containing:
    *   - The language ID. TODO: !!!
    *   - A tensor containing a padded batch of sentences consisting of word IDs.
    *   - A tensor containing the sentence lengths for the aforementioned padded batch.
    */
  protected val input      = Input((INT32, INT32), (Shape(-1, -1), Shape(-1)))
  protected val trainInput = Input((INT32, INT32), (Shape(-1, -1), Shape(-1)))

  // TODO: Make this configurable.
  val parametersManager: ParametersManager = {
    DefaultParametersManager(tf.VarianceScalingInitializer(
      1.0f,
      tf.VarianceScalingInitializer.FanAverageScalingMode,
      tf.VarianceScalingInitializer.UniformDistribution))
  }

  protected val estimator: tf.learn.Estimator[
      (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape), (Output, Output),
      ((Tensor, Tensor), (Tensor, Tensor)), ((Output, Output), (Output, Output)),
      ((DataType, DataType), (DataType, DataType)), ((Shape, Shape), (Shape, Shape)),
      ((Output, Output), (Output, Output))] = tf.createWithNameScope(name) {
    val model = learn.Model.supervised(
      input = input,
      layer = inferLayer,
      trainLayer = trainLayer,
      trainInput = trainInput,
      loss = lossLayer,
      optimizer = optConfig.optimizer,
      clipGradients = tf.learn.ClipGradientsByGlobalNorm(optConfig.maxGradNorm),
      colocateGradientsWithOps = optConfig.colocateGradientsWithOps)
    val summariesDir = config.env.workingDir.resolve("summaries")

    // Create estimator hooks
    var hooks = Set[tf.learn.Hook](
      // tf.learn.LossLogger(trigger = tf.learn.StepHookTrigger(1)),
      tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = StepHookTrigger(100)),
      tf.learn.SummarySaver(summariesDir, StepHookTrigger(config.summarySteps)),
      tf.learn.CheckpointSaver(config.env.workingDir, StepHookTrigger(config.checkpointSteps)))

    // Add logging hooks
    if (logConfig.logLossSteps > 0)
      hooks += TrainingLogger(log = true, trigger = StepHookTrigger(logConfig.logLossSteps))
    if (logConfig.logTrainEvalSteps > 0 && trainEvalDataset != null)
      hooks += tf.learn.Evaluator(
        log = true, summariesDir, trainEvalDataset, Seq(BLEU()), StepHookTrigger(logConfig.logTrainEvalSteps),
        triggerAtEnd = true, name = "TrainEvaluator")
    if (logConfig.logDevEvalSteps > 0 && devEvalDataset != null)
      hooks += tf.learn.Evaluator(
        log = true, summariesDir, devEvalDataset, Seq(BLEU()), StepHookTrigger(logConfig.logDevEvalSteps),
        triggerAtEnd = true, name = "DevEvaluator")
    if (logConfig.logTestEvalSteps > 0 && testEvalDataset != null)
      hooks += tf.learn.Evaluator(
        log = true, summariesDir, testEvalDataset, Seq(BLEU()), StepHookTrigger(logConfig.logTestEvalSteps),
        triggerAtEnd = true, name = "TestEvaluator")

    val estimatorConfig = tf.learn.Configuration(Some(config.env.workingDir), randomSeed = config.env.randomSeed)

    // Create estimator
    tf.learn.InMemoryEstimator(model, estimatorConfig, trainHooks = hooks)
  }

  def train(dataset: () => TFBilingualDataset, stopCriteria: StopCriteria): Unit = {
    estimator.train(dataset, stopCriteria)
  }

  def infer(dataset: () => TFMonolingualDataset): Iterator[((Tensor, Tensor), (Tensor, Tensor))] = {
    estimator.infer(dataset)
  }

  def evaluate(
      dataset: () => TFBilingualDataset,
      metrics: Seq[MTMetric],
      maxSteps: Long = -1L,
      saveSummaries: Boolean = true,
      name: String = null
  ): Seq[Tensor] = {
    estimator.evaluate(dataset, metrics, maxSteps, saveSummaries, name)
  }

  protected def trainLayer: Layer[((Output, Output), (Output, Output)), (Output, Output)] = {
    new Layer[((Output, Output), (Output, Output)), (Output, Output)](name) {
      override val layerType: String = "TrainLayer"

      override protected def _forward(
          input: ((Output, Output), (Output, Output)),
          mode: Mode
      ): (Output, Output) = {
        val state = tf.createWithVariableScope("Encoder")(encoder(input._1, mode))
        val output = tf.createWithVariableScope("Decoder")(decoder(Some(input._2), Some(state), mode))
        output
      }
    }
  }

  protected def inferLayer: Layer[(Output, Output), (Output, Output)] = {
    new Layer[(Output, Output), (Output, Output)](name) {
      override val layerType: String = "InferLayer"

      override protected def _forward(input: (Output, Output), mode: Mode): (Output, Output) = {
        // TODO: The following line is weirdly needed in order to properly initialize the lookup tables.
        srcVocabulary.lookupTable()
        tgtVocabulary.lookupTable()
        val state = tf.createWithVariableScope("Encoder")(encoder(input, mode))
        val output = tf.createWithVariableScope("Decoder")(decoder(None, Some(state), mode))
        output
      }
    }
  }

  protected def lossLayer: Layer[((Output, Output), (Output, Output)), Output] = {
    new Layer[((Output, Output), (Output, Output)), Output](name) {
      override val layerType: String = "Loss"

      override protected def _forward(
          input: ((Output, Output), (Output, Output)),
          mode: Mode
      ): Output = tf.createWithNameScope("Loss") {
        // TODO: Handle this shift more efficiently.
        // Shift the target sequence one step backward so the decoder is evaluated based using the correct previous
        // word used as input, rather than the previous predicted word.
        val tgtEosId = tgtVocabulary.lookupTable().lookup(tf.constant(dataConfig.endOfSequenceToken)).cast(INT32)
        val tgtSequence = tf.concatenate(Seq(
          input._2._1,
          tf.fill(INT32, tf.stack(Seq(tf.shape(input._2._1)(0), 1)))(tgtEosId)), axis = 1)
        val tgtSequenceLength = input._2._2 + 1
        val lossValue = loss(input._1._1, tgtSequence, tgtSequenceLength)
        tf.summary.scalar("Loss", lossValue)
        lossValue
      }
    }
  }

  /**
    *
    * @param  input   Tuple containing two tensors:
    *                 - `INT32` tensor with shape `[batchSize, inputLength]`, containing the sentence word IDs.
    *                 - `INT32` tensor with shape `[batchSize]`, containing the sequence lengths.
    * @param  mode    Current learning mode (e.g., training or evaluation).
    * @return   Tuple containing two tensors:
    *           - Encoder output, with shape `[batchSize, inputLength, hiddenSize]`.
    *           - Encoder-decoder attention bias and mask weights, with shape `[batchSize, inputLength]`.
    */
  protected def encoder(input: (Output, Output), mode: Mode): S

  /**
    *
    * @param input
    * @param state
    * @param mode
    * @return Tensor with shape `[batchSize, length, 1, hiddenSize]`.
    */
  protected def decoder(input: Option[(Output, Output)], state: Option[S], mode: Mode): (Output, Output)

  protected def loss(predictedSequences: Output, targetSequences: Output, targetSequenceLengths: Output): Output = {
    val (lossSum, weightsSum) = Common.paddedCrossEntropy(
      predictedSequences, targetSequences, targetSequenceLengths, config.labelSmoothing, timeMajor = config.timeMajor)
    lossSum / tf.size(targetSequenceLengths).cast(FLOAT32)
  }
}

object Model {
  class Config protected (
      val env: Environment,
      val labelSmoothing: Float,
      val timeMajor: Boolean,
      val summarySteps: Int,
      val checkpointSteps: Int)

  object Config {
    def apply(
        env: Environment,
        labelSmoothing: Float = 0.0f,
        timeMajor: Boolean = false,
        summarySteps: Int = 100,
        checkpointSteps: Int = 1000
    ): Config = {
      new Config(env, labelSmoothing, timeMajor, summarySteps, checkpointSteps)
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
      val logTrainEvalSteps: Int,
      val logDevEvalSteps: Int,
      val logTestEvalSteps: Int)

  object LogConfig {
    def apply(
        logLossSteps: Int = 100,
        logEvalBatchSize: Int = 512,
        logTrainEvalSteps: Int = 1000,
        logDevEvalSteps: Int = 1000,
        logTestEvalSteps: Int = 1000
    ): LogConfig = {
      new LogConfig(logLossSteps, logEvalBatchSize, logTrainEvalSteps, logDevEvalSteps, logTestEvalSteps)
    }
  }
}
