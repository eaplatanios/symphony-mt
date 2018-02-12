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

import org.platanios.symphony.mt.{Environment, Language, LogConfig}
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.evaluation.{BLEU, MTMetric}
import org.platanios.symphony.mt.models.hooks.TrainingLogger
import org.platanios.symphony.mt.models.rnn.{Cell, RNNDecoder, RNNEncoder}
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api.learn.{Mode, StopCriteria}
import org.platanios.tensorflow.api.learn.hooks.StepHookTrigger
import org.platanios.tensorflow.api.learn.layers.{Input, Layer}
import org.platanios.tensorflow.api.learn.layers.rnn.cell.{DeviceWrapper, DropoutWrapper, MultiCell, ResidualWrapper}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.training.optimizers.{GradientDescent, Optimizer}
import org.platanios.tensorflow.api.ops.training.optimizers.decay.{Decay, ExponentialDecay}

/**
  * @author Emmanouil Antonios Platanios
  */
class StateBasedModel[S, SS](
    override val name: String = "Model",
    override val srcLang: Language,
    override val srcVocab: Vocabulary,
    override val tgtLang: Language,
    override val tgtVocab: Vocabulary,
    val config: StateBasedModel.Config[S, SS],
    override val trainEvalDataset: () => MTTrainDataset = null,
    override val devEvalDataset: () => MTTrainDataset = null,
    override val testEvalDataset: () => MTTrainDataset = null,
    override val dataConfig: DataConfig = DataConfig(),
    override val logConfig: LogConfig = LogConfig()
)(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) extends Model(name, srcLang, srcVocab, tgtLang, tgtVocab) {
  // Create the input and the train input parts of the model.
  protected val input      = Input((INT32, INT32), (Shape(-1, -1), Shape(-1)))
  protected val trainInput = Input((INT32, INT32), (Shape(-1, -1), Shape(-1)))

  protected val estimator: tf.learn.Estimator[
      (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape), (Output, Output),
      ((Tensor, Tensor), (Tensor, Tensor)), ((Output, Output), (Output, Output)),
      ((DataType, DataType), (DataType, DataType)), ((Shape, Shape), (Shape, Shape)),
      ((Output, Output), (Output, Output))] = tf.createWithNameScope(name) {
    val model = learn.Model(
      input = input,
      layer = inferLayer,
      trainLayer = trainLayer,
      trainInput = trainInput,
      loss = lossLayer,
      optimizer = optimizer,
      clipGradients = tf.learn.ClipGradientsByGlobalNorm(config.maxGradNorm),
      colocateGradientsWithOps = config.colocateGradientsWithOps)
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

    // Create estimator
    tf.learn.InMemoryEstimator(
      model,
      tf.learn.Configuration(Some(config.env.workingDir), randomSeed = config.env.randomSeed),
      trainHooks = hooks)
  }

  private final def trainLayer: Layer[((Output, Output), (Output, Output)), (Output, Output)] = {
    new Layer[((Output, Output), (Output, Output)), (Output, Output)](name) {
      override val layerType: String = "TrainLayer"

      override protected def _forward(
          input: ((Output, Output), (Output, Output)),
          mode: Mode
      ): (Output, Output) = {
        // TODO: !!! I need to fix this repetition in TensorFlow for Scala.
        val encTuple = tf.createWithVariableScope("Encoder") {
          tf.learn.variableScope("Encoder") {
            config.encoder.create(config.env, input._1._1, input._1._2, srcVocab, mode)
          }
        }
        val decTuple = tf.createWithVariableScope("Decoder") {
          tf.learn.variableScope("Decoder") {
            // TODO: Handle this shift more efficiently.
            // Shift the target sequence one step forward so the decoder learns to output the next word.
            val tgtBosId = tgtVocab.lookupTable().lookup(tf.constant(dataConfig.beginOfSequenceToken)).cast(INT32)
            val tgtSequence = tf.concatenate(Seq(
              tf.fill(INT32, tf.stack(Seq(tf.shape(input._2._1)(0), 1)))(tgtBosId),
              input._2._1), axis = 1)
            val tgtSequenceLength = input._2._2 + 1
            config.decoder.create(config.env, encTuple, input._1._2, tgtVocab, dataConfig.tgtMaxLength,
              dataConfig.beginOfSequenceToken, dataConfig.endOfSequenceToken, tgtSequence, tgtSequenceLength, mode)
          }
        }
        (decTuple.sequences, decTuple.sequenceLengths)
      }
    }
  }

  private final def inferLayer: Layer[(Output, Output), (Output, Output)] = {
    new Layer[(Output, Output), (Output, Output)](name) {
      override val layerType: String = "InferLayer"

      override protected def _forward(input: (Output, Output), mode: Mode): (Output, Output) = {
        // TODO: The following line is weirdly needed in order to properly initialize the lookup table.
        srcVocab.lookupTable()

        val encTuple = tf.createWithVariableScope("Encoder") {
          tf.learn.variableScope("Encoder") {
            config.encoder.create(config.env, input._1, input._2, srcVocab, mode)
          }
        }
        val decTuple = tf.createWithVariableScope("Decoder") {
          tf.learn.variableScope("Decoder") {
            config.decoder.create(
              config.env, encTuple, input._2, tgtVocab, dataConfig.tgtMaxLength,
              dataConfig.beginOfSequenceToken, dataConfig.endOfSequenceToken, null, null, mode)
          }
        }
        // Make sure the outputs are of shape [batchSize, time] or [beamWidth, batchSize, time]
        // when using beam search.
        val outputSequence = {
          if (config.timeMajor)
            decTuple.sequences.transpose()
          else if (decTuple.sequences.rank == 3)
            decTuple.sequences.transpose(Tensor(2, 0, 1))
          else
            decTuple.sequences
        }
        (outputSequence(---, 0 :: -1), decTuple.sequenceLengths - 1)
      }
    }
  }

  private final def lossLayer: Layer[((Output, Output), (Output, Output)), Output] = {
    new Layer[((Output, Output), (Output, Output)), Output](name) {
      override val layerType: String = "Loss"

      override protected def _forward(
          input: ((Output, Output), (Output, Output)),
          mode: Mode
      ): Output = tf.createWithNameScope("Loss") {
        // TODO: Handle this shift more efficiently.
        // Shift the target sequence one step backward so the decoder is evaluated based using the correct previous
        // word used as input, rather than the previous predicted word.
        val tgtEosId = tgtVocab.lookupTable().lookup(tf.constant(dataConfig.endOfSequenceToken)).cast(INT32)
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

  protected def loss(predictedSequences: Output, targetSequences: Output, targetSequenceLengths: Output): Output = {
    val maxTime = tf.shape(targetSequences)(1)
    val transposedTargetSequences = if (config.timeMajor) targetSequences.transpose() else targetSequences
    val crossEntropy = tf.sparseSoftmaxCrossEntropy(predictedSequences, transposedTargetSequences)
    val weights = tf.sequenceMask(targetSequenceLengths, maxTime, predictedSequences.dataType)
    val transposedWeights = if (config.timeMajor) weights.transpose() else weights
    tf.sum(crossEntropy * transposedWeights) / tf.size(targetSequenceLengths).cast(FLOAT32)
  }

  protected def optimizer: tf.train.Optimizer = {
    val decay = ExponentialDecay(
      config.learningRateDecayRate,
      config.learningRateDecaySteps,
      staircase = true,
      config.learningRateDecayStartStep)
    config.optimizer(config.learningRateInitial, decay)
  }

  override def train(dataset: () => MTTrainDataset, stopCriteria: StopCriteria): Unit = {
    estimator.train(dataset, stopCriteria)
  }

  override def infer(dataset: () => MTInferDataset): Iterator[((Tensor, Tensor), (Tensor, Tensor))] = {
    estimator.infer(dataset)
  }

  override def evaluate(
      dataset: () => MTTrainDataset,
      metrics: Seq[MTMetric],
      maxSteps: Long = -1L,
      saveSummaries: Boolean = true,
      name: String = null
  ): Seq[Tensor] = {
    estimator.evaluate(dataset, metrics, maxSteps, saveSummaries, name)
  }
}

object StateBasedModel {
  def apply[S, SS](
      name: String = "Model",
      srcLang: Language,
      srcVocab: Vocabulary,
      tgtLang: Language,
      tgtVocab: Vocabulary,
      config: StateBasedModel.Config[S, SS],
      trainEvalDataset: () => MTTrainDataset = null,
      devEvalDataset: () => MTTrainDataset = null,
      testEvalDataset: () => MTTrainDataset = null,
      dataConfig: DataConfig = DataConfig(),
      logConfig: LogConfig = LogConfig()
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): StateBasedModel[S, SS] = {
    new StateBasedModel[S, SS](
      name, srcLang, srcVocab, tgtLang, tgtVocab, config,
      trainEvalDataset, devEvalDataset, testEvalDataset,
      dataConfig, logConfig)(evS, evSDropout)
  }

  class Config[S, SS] protected (
      val env: Environment,
      // Model
      val encoder: RNNEncoder[S, SS],
      val decoder: RNNDecoder[S, SS],
      val timeMajor: Boolean = false,
      // Training
      val maxGradNorm: Float = 5.0f,
      val optimizer: (Float, Decay) => Optimizer = GradientDescent(_, _, learningRateSummaryTag = "LearningRate"),
      val learningRateInitial: Float = 1.0f,
      val learningRateDecayRate: Float = 1.0f,
      val learningRateDecaySteps: Int = 10000,
      val learningRateDecayStartStep: Int = 0,
      val colocateGradientsWithOps: Boolean = true,
      val summarySteps: Int = 100,
      val checkpointSteps: Int = 1000)

  object Config {
    def apply[S, SS](
        env: Environment,
        // Model
        encoder: RNNEncoder[S, SS],
        decoder: RNNDecoder[S, SS],
        timeMajor: Boolean = false,
        // Training
        maxGradNorm: Float = 5.0f,
        optimizer: (Float, Decay) => Optimizer = GradientDescent(_, _, learningRateSummaryTag = "LearningRate"),
        learningRateInitial: Float = 1.0f,
        learningRateDecayRate: Float = 1.0f,
        learningRateDecaySteps: Int = 10000,
        learningRateDecayStartStep: Int = 0,
        colocateGradientsWithOps: Boolean = true,
        summarySteps: Int = 100,
        checkpointSteps: Int = 1000
    ): Config[S, SS] = {
      new Config[S, SS](
        env, encoder, decoder, timeMajor, maxGradNorm, optimizer, learningRateInitial, learningRateDecayRate,
        learningRateDecaySteps, learningRateDecayStartStep, colocateGradientsWithOps, summarySteps, checkpointSteps)
    }
  }

  private[models] def embeddings(
      dataType: DataType, srcSize: Int, numUnits: Int, name: String = "Embeddings"): Variable = {
    val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
    tf.variable(name, dataType, Shape(srcSize, numUnits), embeddingsInitializer)
  }

  private[this] def device(layerIndex: Int, numGPUs: Int = 0, firstGPU: Int = 0): String = {
    if (numGPUs - firstGPU <= 0)
      "/device:CPU:0"
    else
      s"/device:GPU:${firstGPU + (layerIndex % (numGPUs - firstGPU))}"
  }

  private[models] def cell[S, SS](
      cellCreator: Cell[S, SS],
      numUnits: Int,
      dataType: DataType,
      dropout: Option[Float] = None,
      residualFn: Option[(Output, Output) => Output] = None,
      device: Option[String] = None,
      seed: Option[Int] = None,
      name: String
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): tf.learn.RNNCell[Output, Shape, S, SS] = tf.learn.variableScope(name) {
    var createdCell = cellCreator.create(name, numUnits, dataType)
    createdCell = dropout.map(p => DropoutWrapper("Dropout", createdCell, 1.0f - p, seed = seed)).getOrElse(createdCell)
    createdCell = residualFn.map(ResidualWrapper("Residual", createdCell, _)).getOrElse(createdCell)
    createdCell = device.map(DeviceWrapper("Device", createdCell, _)).getOrElse(createdCell)
    createdCell
  }

  private[models] def cells[S, SS](
      cellCreator: Cell[S, SS],
      numUnits: Int,
      dataType: DataType,
      numLayers: Int,
      numResidualLayers: Int,
      dropout: Option[Float] = None,
      residualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output),
      baseGPU: Int = 0,
      numGPUs: Int = 0,
      firstGPU: Int = 0,
      seed: Option[Int] = None,
      name: String
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): Seq[tf.learn.RNNCell[Output, Shape, S, SS]] = tf.learn.variableScope(name) {
    (0 until numLayers).map(i => {
      cell(
        cellCreator, numUnits, dataType, dropout, if (i >= numLayers - numResidualLayers) residualFn else None,
        Some(device(i + baseGPU, numGPUs, firstGPU)), seed, s"Cell$i")
    })
  }

  private[models] def multiCell[S, SS](
      cellCreator: Cell[S, SS],
      numUnits: Int,
      dataType: DataType,
      numLayers: Int,
      numResidualLayers: Int,
      dropout: Option[Float] = None,
      residualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output),
      baseGPU: Int = 0,
      numGPUs: Int = 0,
      firstGPU: Int = 0,
      seed: Option[Int] = None,
      name: String
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): tf.learn.RNNCell[Output, Shape, Seq[S], Seq[SS]] = {
    MultiCell(name, cells(
      cellCreator, numUnits, dataType, numLayers, numResidualLayers, dropout,
      residualFn, baseGPU, numGPUs, firstGPU, seed, name))
  }
}
