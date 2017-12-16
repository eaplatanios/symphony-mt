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

import org.platanios.symphony.mt.core.Language
import org.platanios.symphony.mt.core.hooks.PerplexityLogger
import org.platanios.symphony.mt.data.Datasets.MTTextLinesDataset
import org.platanios.symphony.mt.metrics.BLEUTensorFlow
import org.platanios.tensorflow.api
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.{Mode, StopCriteria}
import org.platanios.tensorflow.api.learn.hooks.StepHookTrigger
import org.platanios.tensorflow.api.learn.layers.{Layer, LayerInstance}
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.training.optimizers.decay.ExponentialDecay
import org.platanios.tensorflow.api.types.DataType

/**
  * @author Emmanouil Antonios Platanios
  */
class GNMTModel[S, SS](
    override val configuration: Configuration[S, SS],
    val srcLanguage: Language,
    val tgtLanguage: Language,
    val srcTrainDataset: MTTextLinesDataset,
    val tgtTrainDataset: MTTextLinesDataset,
    val srcDevDataset: MTTextLinesDataset = null,
    val tgtDevDataset: MTTextLinesDataset = null,
    val srcTestDataset: MTTextLinesDataset = null,
    val tgtTestDataset: MTTextLinesDataset = null,
    override val name: String = "GNMTModel"
)(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) extends Model[S, SS] {
  // Create the input and the train input parts of the model.
  protected val input      = tf.learn.Input((INT32, INT32), (Shape(-1, -1), Shape(-1)))
  protected val trainInput = tf.learn.Input((INT32, INT32, INT32), (Shape(-1, -1), Shape(-1, -1), Shape(-1)))

  protected val encoder: Encoder[S, SS] = GNMTEncoder(configuration, "Encoder")(evS, evSDropout)
  protected val decoder: Decoder[S, SS] = GNMTDecoder(configuration, "Decoder")(evS, evSDropout)

  protected val srcVocab: () => api.tf.LookupTable = srcLanguage.vocabulary
  protected val tgtVocab: () => api.tf.LookupTable = tgtLanguage.vocabulary

  protected val estimator: tf.learn.Estimator[
      (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape), (Output, Output),
      ((Tensor, Tensor), (Tensor, Tensor, Tensor)), ((Output, Output), (Output, Output, Output)),
      ((DataType, DataType), (DataType, DataType, DataType)), ((Shape, Shape), (Shape, Shape, Shape)),
      ((Output, Output), (Output, Output, Output))] = tf.createWithNameScope(name) {
    val model = tf.learn.Model(
      input = input,
      layer = inferLayer,
      trainLayer = trainLayer,
      trainInput = trainInput,
      loss = lossLayer,
      optimizer = optimizer,
      clipGradients = tf.learn.ClipGradientsByGlobalNorm(configuration.trainMaxGradNorm),
      colocateGradientsWithOps = configuration.trainColocateGradientsWithOps)
    val summariesDir = configuration.workingDir.resolve("summaries")
    val tensorBoardConfig = {
      if (configuration.trainLaunchTensorBoard)
        tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 1)
      else
        null
    }
    var hooks = Set[tf.learn.Hook](
      tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = StepHookTrigger(100)),
      tf.learn.SummarySaver(summariesDir, StepHookTrigger(configuration.trainSummarySteps)),
      tf.learn.CheckpointSaver(configuration.workingDir, StepHookTrigger(configuration.trainCheckpointSteps)))
    if (configuration.logLossSteps > 0)
      hooks += PerplexityLogger(log = true, trigger = StepHookTrigger(configuration.logLossSteps))
    if (configuration.logTrainEvalSteps > 0)
      hooks += tf.learn.Evaluator(
        log = true, summariesDir,
        () => createTrainDataset(
          srcTrainDataset, tgtTrainDataset, srcVocab, tgtVocab, configuration.logEvalBatchSize, repeat = false, 1),
        Seq(BLEUTensorFlow()), StepHookTrigger(configuration.logTrainEvalSteps),
        triggerAtEnd = true, name = "Train Evaluation")
    if (configuration.logDevEvalSteps > 0 && srcDevDataset != null && tgtDevDataset != null)
      hooks += tf.learn.Evaluator(
        log = true, summariesDir,
        () => createTrainDataset(
          srcDevDataset, tgtDevDataset, srcVocab, tgtVocab, configuration.logEvalBatchSize, repeat = false, 1),
        Seq(BLEUTensorFlow()), StepHookTrigger(configuration.logDevEvalSteps),
        triggerAtEnd = true, name = "Dev Evaluation")
    if (configuration.logTestEvalSteps > 0 && srcTestDataset != null && tgtTestDataset != null)
      hooks += tf.learn.Evaluator(
        log = true, summariesDir,
        () => createTrainDataset(
          srcTestDataset, tgtTestDataset, srcVocab, tgtVocab, configuration.logEvalBatchSize, repeat = false, 1),
        Seq(BLEUTensorFlow()), StepHookTrigger(configuration.logTestEvalSteps),
        triggerAtEnd = true, name = "Test Evaluation")
    tf.learn.InMemoryEstimator(
      model, tf.learn.Configuration(Some(configuration.workingDir), randomSeed = configuration.randomSeed),
      StopCriteria(Some(configuration.trainNumSteps)), hooks, tensorBoardConfig = tensorBoardConfig)
  }

  protected def trainLayer: Layer[((Output, Output), (Output, Output, Output)), (Output, Output)] = {
    new Layer[((Output, Output), (Output, Output, Output)), (Output, Output)]("GNMTModelTrainLayer") {
      override val layerType: String = "GNMTModelTrainLayer"

      override def forward(
          input: ((Output, Output), (Output, Output, Output)),
          mode: Mode
      ): LayerInstance[((Output, Output), (Output, Output, Output)), (Output, Output)] = {
        val encLayerInstance = encoder.layer(input._1, mode)
        val decLayerInstance = decoder.trainLayer((input._2, encLayerInstance.output), mode)
        LayerInstance(
          input, decLayerInstance.output,
          encLayerInstance.trainableVariables ++ decLayerInstance.trainableVariables,
          encLayerInstance.nonTrainableVariables ++ decLayerInstance.nonTrainableVariables)
      }
    }
  }

  protected def inferLayer: Layer[(Output, Output), (Output, Output)] = {
    new Layer[(Output, Output), (Output, Output)]("GNMTModelInferLayer") {
      override val layerType: String = "GNMTModelInferLayer"

      override def forward(input: (Output, Output), mode: Mode): LayerInstance[(Output, Output), (Output, Output)] = {
        val encLayerInstance = encoder.layer(input, mode)
        val decLayerInstance = decoder.inferLayer((input, encLayerInstance.output), mode)
        LayerInstance(
          input, decLayerInstance.output,
          encLayerInstance.trainableVariables ++ decLayerInstance.trainableVariables,
          encLayerInstance.nonTrainableVariables ++ decLayerInstance.nonTrainableVariables)
      }
    }
  }

  protected def lossLayer: Layer[((Output, Output), (Output, Output, Output)), Output] = {
    new tf.learn.Layer[((Output, Output), (Output, Output, Output)), Output]("GNMTModelLoss") {
      override val layerType: String = "GNMTModelLoss"

      override def forward(
          input: ((Output, Output), (Output, Output, Output)),
          mode: Mode
      ): LayerInstance[((Output, Output), (Output, Output, Output)), Output] = {
        val loss = tf.sum(tf.sequenceLoss(
          input._1._1, input._2._2,
          weights = tf.sequenceMask(input._1._2, tf.shape(input._1._1)(1), dataType = input._1._1.dataType),
          averageAcrossTimeSteps = false, averageAcrossBatch = true))
        tf.summary.scalar("Loss", loss)
        LayerInstance(input, loss)
      }
    }
  }

  protected def optimizer: tf.train.Optimizer = {
    val decay = ExponentialDecay(
      configuration.trainLearningRateDecayRate,
      configuration.trainLearningRateDecaySteps,
      staircase = true,
      configuration.trainLearningRateDecayStartStep)
    configuration.trainOptimizer(configuration.trainLearningRateInitial, decay)
  }

  override def train(stopCriteria: StopCriteria = StopCriteria(Some(configuration.trainNumSteps))): Unit = {
    val trainDataset = () => createTrainDataset(
      srcTrainDataset, tgtTrainDataset, srcVocab, tgtVocab, configuration.trainBatchSize, repeat = true,
      configuration.dataNumBuckets)
    estimator.train(trainDataset, stopCriteria)
  }
}

object GNMTModel {
  def apply[S, SS](
      configuration: Configuration[S, SS],
      srcLanguage: Language,
      tgtLanguage: Language,
      srcTrainDataset: MTTextLinesDataset,
      tgtTrainDataset: MTTextLinesDataset,
      srcDevDataset: MTTextLinesDataset = null,
      tgtDevDataset: MTTextLinesDataset = null,
      srcTestDataset: MTTextLinesDataset = null,
      tgtTestDataset: MTTextLinesDataset = null,
      name: String = "GNMTModel"
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): GNMTModel[S, SS] = {
    new GNMTModel[S, SS](
      configuration,
      srcLanguage, tgtLanguage,
      srcTrainDataset, tgtTrainDataset,
      srcDevDataset, tgtDevDataset,
      srcTestDataset, tgtTestDataset)
  }

  private[this] def device(layerIndex: Int, numGPUs: Int = 0): String = {
    if (numGPUs == 0)
      "/device:CPU:0"
    else
      s"/device:GPU:${layerIndex % numGPUs}"
  }

  def cell[S, SS](
      cellCreator: Cell[S, SS],
      numUnits: Int,
      dataType: DataType,
      dropout: Option[Float] = None,
      residualFn: Option[(Output, Output) => Output] = None,
      device: Option[String] = None,
      seed: Option[Int] = None,
      name: String = "RNNCell"
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): tf.learn.RNNCell[Output, Shape, S, SS] = tf.createWithNameScope(name) {
    var createdCell = cellCreator.create(numUnits, dataType, name)
    createdCell = dropout.map(p => tf.learn.DropoutWrapper(createdCell, 1.0f - p, seed = seed)).getOrElse(createdCell)
    createdCell = residualFn.map(tf.learn.ResidualWrapper(createdCell, _)).getOrElse(createdCell)
    createdCell = device.map(tf.learn.DeviceWrapper(createdCell, _)).getOrElse(createdCell)
    createdCell
  }

  def cells[S, SS](
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
      name: String = "RNNCells"
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): Seq[tf.learn.RNNCell[Output, Shape, S, SS]] = tf.createWithNameScope(name) {
    (0 until numLayers).map(i => {
      cell(
        cellCreator, numUnits, dataType, dropout, if (i >= numLayers - numResidualLayers) residualFn else None,
        Some(device(i + baseGPU, numGPUs)), seed, s"$name/Cell$i")
    })
  }

  def multiCell[S, SS](
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
      name: String = "RNNMultiCell"
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): tf.learn.RNNCell[Output, Shape, Seq[S], Seq[SS]] = {
    tf.learn.MultiRNNCell(cells(
      cellCreator, numUnits, dataType, numLayers, numResidualLayers, dropout,
      residualFn, baseGPU, numGPUs, seed, name), name)
  }
}
