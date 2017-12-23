/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.symphony.mt.models.rnn

import org.platanios.symphony.mt.{Environment, Language, LogConfig}
import org.platanios.symphony.mt.core.hooks.PerplexityLogger
import org.platanios.symphony.mt.data.{DataConfig, Datasets, Vocabulary}
import org.platanios.symphony.mt.data.Datasets.{MTTextLinesDataset, MTTrainDataset}
import org.platanios.symphony.mt.metrics.BLEUTensorFlow
import org.platanios.symphony.mt.models.{InferConfig, TrainConfig}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn
import org.platanios.tensorflow.api.learn.hooks.StepHookTrigger
import org.platanios.tensorflow.api.learn.layers.rnn.cell._
import org.platanios.tensorflow.api.learn.{Mode, StopCriteria}
import org.platanios.tensorflow.api.learn.layers.{Input, Layer}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.training.optimizers.decay.ExponentialDecay
import org.platanios.tensorflow.api.types.DataType

// TODO: Move embeddings initializer to the configuration.
// TODO: Add support for optimizer schedules (e.g., Adam for first 1000 steps and then SGD with a different learning rate.
// TODO: Customize evaluation metrics, hooks, etc.

/**
  * @author Emmanouil Antonios Platanios
  */
class Model[S, SS](
    val config: Model.Config[S, SS],
    val srcLanguage: Language,
    val tgtLanguage: Language,
    val srcVocabulary: Vocabulary,
    val tgtVocabulary: Vocabulary,
    val srcTrainDataset: MTTextLinesDataset,
    val tgtTrainDataset: MTTextLinesDataset,
    val srcDevDataset: MTTextLinesDataset = null,
    val tgtDevDataset: MTTextLinesDataset = null,
    val srcTestDataset: MTTextLinesDataset = null,
    val tgtTestDataset: MTTextLinesDataset = null,
    val env: Environment = Environment(),
    val rnnConfig: RNNConfig = RNNConfig(),
    val dataConfig: DataConfig = DataConfig(),
    val trainConfig: TrainConfig = TrainConfig(),
    val inferConfig: InferConfig = InferConfig(),
    val logConfig: LogConfig = LogConfig(),
    val name: String = "Model"
)(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) {
  // Create the input and the train input parts of the model.
  protected val input      = Input((INT32, INT32), (Shape(-1, -1), Shape(-1)))
  protected val trainInput = Input((INT32, INT32, INT32), (Shape(-1, -1), Shape(-1, -1), Shape(-1)))

  protected def createSupervisedDataset(
      srcDataset: MTTextLinesDataset,
      tgtDataset: MTTextLinesDataset,
      batchSize: Int,
      repeat: Boolean,
      numBuckets: Int
  ): MTTrainDataset = {
    Datasets.createTrainDataset(
      srcDataset, tgtDataset, srcVocabulary.lookupTable(), tgtVocabulary.lookupTable(), dataConfig, batchSize, repeat,
      env.randomSeed)
  }

  protected def loss(predictedSequences: Output, targetSequences: Output, targetSequenceLengths: Output): Output = {
    val maxTime = tf.shape(targetSequences)(1)
    val transposedTargetSequences = if (rnnConfig.timeMajor) targetSequences.transpose() else targetSequences
    val crossEntropy = tf.sparseSoftmaxCrossEntropy(predictedSequences, transposedTargetSequences)
    val weights = tf.sequenceMask(targetSequenceLengths, maxTime, predictedSequences.dataType)
    val transposedWeights = if (rnnConfig.timeMajor) weights.transpose() else weights
    tf.sum(crossEntropy * transposedWeights) / tf.size(targetSequenceLengths).cast(FLOAT32)
  }

  protected def optimizer: tf.train.Optimizer = {
    val decay = ExponentialDecay(
      trainConfig.learningRateDecayRate,
      trainConfig.learningRateDecaySteps,
      staircase = true,
      trainConfig.learningRateDecayStartStep)
    trainConfig.optimizer(trainConfig.learningRateInitial, decay)
  }

  protected val estimator: tf.learn.Estimator[
      (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape), (Output, Output),
      ((Tensor, Tensor), (Tensor, Tensor, Tensor)), ((Output, Output), (Output, Output, Output)),
      ((DataType, DataType), (DataType, DataType, DataType)), ((Shape, Shape), (Shape, Shape, Shape)),
      ((Output, Output), (Output, Output, Output))] = tf.createWithNameScope(name) {
    val model = learn.Model(
      input = input,
      layer = inferLayer,
      trainLayer = trainLayer,
      trainInput = trainInput,
      loss = lossLayer,
      optimizer = optimizer,
      clipGradients = tf.learn.ClipGradientsByGlobalNorm(trainConfig.maxGradNorm),
      colocateGradientsWithOps = trainConfig.colocateGradientsWithOps)
    val summariesDir = env.workingDir.resolve("summaries")
    val tensorBoardConfig = {
      if (trainConfig.launchTensorBoard)
        tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 1)
      else
        null
    }
    var hooks = Set[tf.learn.Hook](
      // tf.learn.LossLogger(trigger = tf.learn.StepHookTrigger(1)),
      tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = StepHookTrigger(100)),
      tf.learn.SummarySaver(summariesDir, StepHookTrigger(trainConfig.summarySteps)),
      tf.learn.CheckpointSaver(env.workingDir, StepHookTrigger(trainConfig.checkpointSteps)))
    if (logConfig.logLossSteps > 0)
      hooks += PerplexityLogger(log = true, trigger = StepHookTrigger(logConfig.logLossSteps))
    if (logConfig.logTrainEvalSteps > 0 && srcTrainDataset != null && tgtTrainDataset != null)
      hooks += tf.learn.Evaluator(
        log = true, summariesDir,
        () => createSupervisedDataset(srcTrainDataset, tgtTrainDataset, logConfig.logEvalBatchSize, repeat = false, 1),
        Seq(BLEUTensorFlow()), StepHookTrigger(logConfig.logTrainEvalSteps),
        triggerAtEnd = true, name = "TrainEvaluator")
    if (logConfig.logDevEvalSteps > 0 && srcDevDataset != null && tgtDevDataset != null)
      hooks += tf.learn.Evaluator(
        log = true, summariesDir,
        () => createSupervisedDataset(srcDevDataset, tgtDevDataset, logConfig.logEvalBatchSize, repeat = false, 1),
        Seq(BLEUTensorFlow()), StepHookTrigger(logConfig.logDevEvalSteps),
        triggerAtEnd = true, name = "DevEvaluator")
    if (logConfig.logTestEvalSteps > 0 && srcTestDataset != null && tgtTestDataset != null)
      hooks += tf.learn.Evaluator(
        log = true, summariesDir,
        () => createSupervisedDataset(srcTestDataset, tgtTestDataset, logConfig.logEvalBatchSize, repeat = false, 1),
        Seq(BLEUTensorFlow()), StepHookTrigger(logConfig.logTestEvalSteps),
        triggerAtEnd = true, name = "TestEvaluator")
    tf.learn.InMemoryEstimator(
      model, tf.learn.Configuration(Some(env.workingDir), randomSeed = env.randomSeed),
      StopCriteria(Some(trainConfig.numSteps)), hooks, tensorBoardConfig = tensorBoardConfig)
  }

  private final def trainLayer: Layer[((Output, Output), (Output, Output, Output)), (Output, Output)] = {
    new Layer[((Output, Output), (Output, Output, Output)), (Output, Output)](name) {
      override val layerType: String = "TrainLayer"

      override def forward(input: ((Output, Output), (Output, Output, Output)), mode: Mode): (Output, Output) = {
        val encTuple = tf.createWithVariableScope("Encoder") {
          config.encoder.create(input._1._1, input._1._2, mode)
        }
        val decTuple = tf.createWithVariableScope("Decoder") {
          config.decoder.create(encTuple, input._1._2, input._2._1, input._2._3, mode)
        }
        (decTuple.sequences, decTuple.sequenceLengths)
      }
    }
  }

  private final def inferLayer: Layer[(Output, Output), (Output, Output)] = {
    new Layer[(Output, Output), (Output, Output)](name) {
      override val layerType: String = "InferLayer"

      override def forward(input: (Output, Output), mode: Mode): (Output, Output) = {
        // TODO: The following line is weirdly needed in order to properly initialize the lookup table.
        srcVocabulary.lookupTable()

        val encTuple = tf.createWithVariableScope("Encoder") {
          config.encoder.create(input._1, input._2, mode)
        }
        val decTuple = tf.createWithVariableScope("Decoder") {
          config.decoder.create(encTuple, input._2, null, null, mode)
        }
        // Make sure the outputs are of shape [batchSize, time] or [beamWidth, batchSize, time]
        // when using beam search.
        val outputSequence = {
          if (rnnConfig.timeMajor)
            decTuple.sequences.transpose()
          else if (decTuple.sequences.rank == 3)
            decTuple.sequences.transpose(Tensor(2, 0, 1))
          else
            decTuple.sequences
        }
        (outputSequence, decTuple.sequenceLengths)
      }
    }
  }

  private final def lossLayer: Layer[((Output, Output), (Output, Output, Output)), Output] = {
    new Layer[((Output, Output), (Output, Output, Output)), Output](name) {
      override val layerType: String = "Loss"

      override def forward(
          input: ((Output, Output), (Output, Output, Output)),
          mode: Mode
      ): Output = tf.createWithNameScope("Loss") {
        val lossValue = loss(input._1._1, input._2._2, input._2._3)
        tf.summary.scalar("Loss", lossValue)
        lossValue
      }
    }
  }

  def train(stopCriteria: StopCriteria = StopCriteria(Some(trainConfig.numSteps))): Unit = {
    val trainDataset = () => createSupervisedDataset(
      srcTrainDataset, tgtTrainDataset, trainConfig.batchSize, repeat = true, dataConfig.numBuckets)
    estimator.train(trainDataset, stopCriteria)
  }
}

object Model {
  def apply[S, SS](
      config: Model.Config[S, SS],
      srcLanguage: Language,
      tgtLanguage: Language,
      srcVocabulary: Vocabulary,
      tgtVocabulary: Vocabulary,
      srcTrainDataset: MTTextLinesDataset,
      tgtTrainDataset: MTTextLinesDataset,
      srcDevDataset: MTTextLinesDataset = null,
      tgtDevDataset: MTTextLinesDataset = null,
      srcTestDataset: MTTextLinesDataset = null,
      tgtTestDataset: MTTextLinesDataset = null,
      env: Environment = Environment(),
      rnnConfig: RNNConfig = RNNConfig(),
      dataConfig: DataConfig = DataConfig(),
      trainConfig: TrainConfig = TrainConfig(),
      inferConfig: InferConfig = InferConfig(),
      logConfig: LogConfig = LogConfig(),
      name: String = "Model"
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): Model[S, SS] = {
    new Model[S, SS](
      config, srcLanguage, tgtLanguage, srcVocabulary, tgtVocabulary,
      srcTrainDataset, tgtTrainDataset, srcDevDataset, tgtDevDataset, srcTestDataset, tgtTestDataset,
      env, rnnConfig, dataConfig, trainConfig, inferConfig, logConfig, name)(evS, evSDropout)
  }

  class Config[S, SS](val encoder: RNNEncoder[S, SS], val decoder: RNNDecoder[S, SS])

  object Config {
    def apply[S, SS](encoder: RNNEncoder[S, SS], decoder: RNNDecoder[S, SS]): Config[S, SS] = {
      new Config[S, SS](encoder, decoder)
    }
  }

  private[rnn] def embeddings(
      dataType: DataType, srcSize: Int, numUnits: Int, name: String = "Embeddings"): Variable = {
    val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
    tf.variable(name, dataType, Shape(srcSize, numUnits), embeddingsInitializer)
  }

  private[this] def device(layerIndex: Int, numGPUs: Int = 0): String = {
    if (numGPUs == 0)
      "/device:CPU:0"
    else
      s"/device:GPU:${layerIndex % numGPUs}"
  }

  private[rnn] def cell[S, SS](
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
  ): tf.learn.RNNCell[Output, Shape, S, SS] = tf.learn.nameScope(name) {
    var createdCell = cellCreator.create(name, numUnits, dataType)
    createdCell = dropout.map(p => DropoutWrapper("Dropout", createdCell, 1.0f - p, seed = seed)).getOrElse(createdCell)
    createdCell = residualFn.map(ResidualWrapper("Residual", createdCell, _)).getOrElse(createdCell)
    createdCell = device.map(DeviceWrapper("Device", createdCell, _)).getOrElse(createdCell)
    createdCell
  }

  private[rnn] def cells[S, SS](
      cellCreator: Cell[S, SS],
      numUnits: Int,
      dataType: DataType,
      numLayers: Int,
      numResidualLayers: Int,
      dropout: Option[Float] = None,
      residualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output),
      baseGPU: Int = 0,
      numGPUs: Int = 0,
      seed: Option[Int] = None,
      name: String
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): Seq[tf.learn.RNNCell[Output, Shape, S, SS]] = tf.learn.nameScope(name) {
    (0 until numLayers).map(i => {
      cell(
        cellCreator, numUnits, dataType, dropout, if (i >= numLayers - numResidualLayers) residualFn else None,
        Some(device(i + baseGPU, numGPUs)), seed, s"Cell$i")
    })
  }

  private[rnn] def multiCell[S, SS](
      cellCreator: Cell[S, SS],
      numUnits: Int,
      dataType: DataType,
      numLayers: Int,
      numResidualLayers: Int,
      dropout: Option[Float] = None,
      residualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output),
      baseGPU: Int = 0,
      numGPUs: Int = 0,
      seed: Option[Int] = None,
      name: String,
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): tf.learn.RNNCell[Output, Shape, Seq[S], Seq[SS]] = {
    MultiRNNCell(name, cells(
      cellCreator, numUnits, dataType, numLayers, numResidualLayers, dropout,
      residualFn, baseGPU, numGPUs, seed, name))
  }
}
