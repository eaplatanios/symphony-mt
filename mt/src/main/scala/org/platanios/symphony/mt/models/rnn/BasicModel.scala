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
import org.platanios.symphony.mt.data.Datasets.MTTextLinesDataset
import org.platanios.symphony.mt.data.{DataConfig, Vocabulary}
import org.platanios.symphony.mt.models.{InferConfig, TrainConfig}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.rnn.cell.CellInstance
import org.platanios.tensorflow.api.learn.layers.{Layer, LayerInstance}
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.cell
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple

/**
  * @author Emmanouil Antonios Platanios
  */
class BasicModel[S, SS](
    val config: BasicModel.Config[S, SS],
    override val srcLanguage: Language,
    override val tgtLanguage: Language,
    override val srcVocabulary: Vocabulary,
    override val tgtVocabulary: Vocabulary,
    override val srcTrainDataset: MTTextLinesDataset,
    override val tgtTrainDataset: MTTextLinesDataset,
    override val srcDevDataset: MTTextLinesDataset = null,
    override val tgtDevDataset: MTTextLinesDataset = null,
    override val srcTestDataset: MTTextLinesDataset = null,
    override val tgtTestDataset: MTTextLinesDataset = null,
    override val env: Environment = Environment(),
    override val dataConfig: DataConfig = DataConfig(),
    override val trainConfig: TrainConfig = TrainConfig(),
    override val inferConfig: InferConfig = InferConfig(),
    override val logConfig: LogConfig = LogConfig(),
    override val name: String = "BasicModel"
)(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) extends Model[S, SS] {
  override protected def encoder: Layer[(Output, Output), cell.Tuple[Output, Seq[S]]] = {
    new Layer[(Output, Output), Tuple[Output, Seq[S]]](name) {
      override val layerType: String = "BasicEncoder"

      override def forward(
          input: (Output, Output),
          mode: Mode
      ): LayerInstance[(Output, Output), Tuple[Output, Seq[S]]] = {
        val dataType = config.dataType
        tf.createWithNameScope(uniquifiedName) {
          // Keep track of trainable and non-trainable variables
          var trainableVariables = Set.empty[Variable]
          var nonTrainableVariables = Set.empty[Variable]

          // Embeddings
          val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
          val embeddings = variable(
            "EncoderEmbeddings", dataType, Shape(srcVocabulary.size, config.numUnits), embeddingsInitializer)
          val embeddedInput = tf.embeddingLookup(embeddings, input._1)
          trainableVariables += embeddings

          val tuple = config.encoderType match {
            case BasicModel.UnidirectionalEncoder =>
              val uniCell = Model.multiCell(
                config.cell, config.numUnits, dataType, config.numLayers, config.numResLayers, config.dropout,
                config.residualFn, 0, env.numGPUs, env.randomSeed, s"$name/MultiUniCell")
              val uniCellInstance = uniCell.createCell(mode, embeddedInput.shape)
              trainableVariables ++= uniCellInstance.trainableVariables
              nonTrainableVariables ++= uniCellInstance.nonTrainableVariables
              tf.dynamicRNN(
                uniCellInstance.cell, embeddedInput, null, dataConfig.timeMajor, env.parallelIterations,
                env.swapMemory, input._2, s"$uniquifiedName/UnidirectionalLayers")
            case BasicModel.BidirectionalEncoder =>
              require(config.numLayers % 2 == 0, "Layers number must be even for bidirectional encoders.")
              require(config.numResLayers % 2 == 0, "Residual layers number must be even for bidirectional encoders.")
              val biCellFw = Model.multiCell(
                config.cell, config.numUnits, dataType, config.numLayers / 2, config.numResLayers / 2, config.dropout,
                config.residualFn, 0, env.numGPUs, env.randomSeed, s"$name/MultiBiCellFw")
              val biCellBw = Model.multiCell(
                config.cell, config.numUnits, dataType, config.numLayers / 2, config.numResLayers / 2, config.dropout,
                config.residualFn, config.numLayers / 2, env.numGPUs, env.randomSeed, s"$name/MultiBiCellBw")
              val biCellInstanceFw = biCellFw.createCell(mode, embeddedInput.shape)
              val biCellInstanceBw = biCellBw.createCell(mode, embeddedInput.shape)
              val unmergedBiTuple = tf.bidirectionalDynamicRNN(
                biCellInstanceFw.cell, biCellInstanceBw.cell, embeddedInput, null, null, dataConfig.timeMajor,
                env.parallelIterations, env.swapMemory, input._2,
                s"$uniquifiedName/BidirectionalLayers")
              trainableVariables ++= biCellInstanceFw.trainableVariables ++ biCellInstanceBw.trainableVariables
              nonTrainableVariables ++= biCellInstanceFw.nonTrainableVariables ++ biCellInstanceBw.nonTrainableVariables
              Tuple(
                tf.concatenate(Seq(unmergedBiTuple._1.output, unmergedBiTuple._2.output), -1),
                unmergedBiTuple._1.state.map(List(_))
                    .zipAll(unmergedBiTuple._2.state.map(List(_)), Nil, Nil)
                    .flatMap(Function.tupled(_ ::: _)))
          }
          LayerInstance(input, tuple, trainableVariables, nonTrainableVariables)
        }
      }
    }
  }

  override protected def trainDecoder: Layer[((Output, Output, Output), Tuple[Output, Seq[S]]), (Output, Output)] = {
    new Layer[((Output, Output, Output), Tuple[Output, Seq[S]]), (Output, Output)](name) {
      override val layerType: String = "BasicTrainDecoder"

      override def forward(
          input: ((Output, Output, Output), Tuple[Output, Seq[S]]),
          mode: Mode
      ): LayerInstance[((Output, Output, Output), Tuple[Output, Seq[S]]), (Output, Output)] = {
        val dataType = config.dataType
        tf.createWithNameScope(uniquifiedName) {
          // Keep track of trainable and non-trainable variables
          var trainableVariables = Set.empty[Variable]
          var nonTrainableVariables = Set.empty[Variable]

          // Embeddings
          val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
          val embeddings = variable(
            "DecoderEmbeddings", dataType, Shape(tgtVocabulary.size, config.numUnits), embeddingsInitializer)
          val embeddedInput = tf.embeddingLookup(embeddings, input._1._1)
          trainableVariables += embeddings

          // Decoder
          val (output, otherTrainableVariables, otherNonTrainableVariables) = createCellAndDecode(
            input._1._3, input._2.state, embeddings, embeddedInput, input._2.output, variable(_, _, _, _),
            isTrain = true, mode)
          trainableVariables ++= otherTrainableVariables
          nonTrainableVariables ++= otherNonTrainableVariables

          LayerInstance(input, output, trainableVariables, nonTrainableVariables)
        }
      }
    }
  }

  override protected def inferDecoder: Layer[((Output, Output), Tuple[Output, Seq[S]]), (Output, Output)] = {
    new Layer[((Output, Output), Tuple[Output, Seq[S]]), (Output, Output)](name) {
      override val layerType: String = "BasicInferDecoder"

      override def forward(
          input: ((Output, Output), Tuple[Output, Seq[S]]),
          mode: Mode
      ): LayerInstance[((Output, Output), Tuple[Output, Seq[S]]), (Output, Output)] = {
        val dataType = config.dataType
        tf.createWithNameScope(uniquifiedName) {
          // TODO: The following line is weirdly needed in order to properly initialize the lookup table.
          srcVocabulary.lookupTable()
          // Keep track of trainable and non-trainable variables
          var trainableVariables = Set.empty[Variable]
          var nonTrainableVariables = Set.empty[Variable]

          // Embeddings
          val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
          val embeddings = variable(
            "DecoderEmbeddings", dataType, Shape(tgtVocabulary.size, config.numUnits), embeddingsInitializer)
          val embeddedInput = tf.embeddingLookup(embeddings, input._1._1)
          trainableVariables += embeddings

          // Decoder
          val (output, otherTrainableVariables, otherNonTrainableVariables) = createCellAndDecode(
            input._1._2, input._2.state, embeddings, embeddedInput, input._2.output, variable(_, _, _, _),
            isTrain = false, mode)
          trainableVariables ++= otherTrainableVariables
          nonTrainableVariables ++= otherNonTrainableVariables

          LayerInstance(input, output, trainableVariables, nonTrainableVariables)
        }
      }
    }
  }

  protected def createCellAndDecode(
      inputSequenceLengths: Output,
      inputState: Seq[S],
      embeddings: Variable,
      embeddedInput: Output,
      encoderOutput: Output,
      variableFn: (String, DataType, Shape, tf.VariableInitializer) => Variable,
      isTrain: Boolean,
      mode: Mode
  ): ((Output, Output), Set[Variable], Set[Variable]) = {
    val dataType = config.dataType
    config.decoderAttention match {
      case None =>
        val cellInstance = Model.multiCell(
          config.cell, config.numUnits, dataType, config.numLayers, config.numResLayers, config.dropout,
          config.residualFn, 0, env.numGPUs, env.randomSeed, s"$name/MultiCell").createCell(mode, embeddedInput.shape)
        (decode(
          inputSequenceLengths, inputState, embeddings, embeddedInput, cellInstance, variableFn, isTrain, mode),
            cellInstance.trainableVariables, cellInstance.nonTrainableVariables)
      case Some(a) =>
        // Ensure memory is batch-major
        val memory = if (dataConfig.timeMajor) encoderOutput.transpose(Tensor(1, 0, 2)) else encoderOutput
        val memoryWeights = variableFn("MemoryWeights", dataType, Shape(memory.shape(-1), config.numUnits), null)
        val attention = a.create(memory, memoryWeights.value, inputSequenceLengths, "Attention")
        val attentionWeights = variableFn(
          "AttentionWeights", attention.dataType,
          Shape(config.numUnits + memory.shape(-1), config.numUnits), null)
        val cell = Model.multiCell(
          config.cell, config.numUnits, dataType, config.numLayers, config.numResLayers, config.dropout,
          config.residualFn, 0, env.numGPUs, env.randomSeed, s"$name/MultiCell")
        val cellInstance = cell.createCell(mode, Shape(embeddedInput.shape(-1) + config.numUnits))
        val attentionCell = tf.AttentionWrapperCell(
          cellInstance.cell, Seq(attention), Seq(attentionWeights.value),
          outputAttention = config.decoderOutputAttention)
        val attentionCellInstance = CellInstance(
          cell = attentionCell, trainableVariables = cellInstance.trainableVariables + attentionWeights,
          nonTrainableVariables = cellInstance.nonTrainableVariables)
        (decode(
          inputSequenceLengths, attentionCell.initialState(inputState, dataType), embeddings, embeddedInput,
          attentionCellInstance, variableFn, isTrain, mode),
            cellInstance.trainableVariables + attentionWeights,
            cellInstance.nonTrainableVariables)
    }
  }
}

object BasicModel {
  def apply[S, SS](
      config: Config[S, SS],
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
      dataConfig: DataConfig = DataConfig(),
      trainConfig: TrainConfig = TrainConfig(),
      inferConfig: InferConfig = InferConfig(),
      logConfig: LogConfig = LogConfig(),
      name: String = "BasicModel"
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): BasicModel[S, SS] = {
    new BasicModel[S, SS](
      config, srcLanguage, tgtLanguage, srcVocabulary, tgtVocabulary, srcTrainDataset, tgtTrainDataset, srcDevDataset,
      tgtDevDataset, srcTestDataset, tgtTestDataset, env, dataConfig, trainConfig, inferConfig, logConfig, name)
  }

  sealed trait EncoderType
  case object UnidirectionalEncoder extends EncoderType
  case object BidirectionalEncoder extends EncoderType

  case class Config[S, SS](
      cell: Cell[S, SS],
      numUnits: Int,
      numLayers: Int = 1,
      numResLayers: Int = 0,
      encoderType: BasicModel.EncoderType = BasicModel.UnidirectionalEncoder,
      dataType: DataType = FLOAT32,
      dropout: Option[Float] = None,
      residualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output),
      decoderAttention: Option[Attention] = None,
      decoderOutputAttention: Boolean = false)
}
