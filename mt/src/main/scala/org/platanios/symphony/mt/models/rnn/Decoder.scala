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

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.{EVALUATION, INFERENCE, Mode}
import org.platanios.tensorflow.api.learn.layers.rnn.cell.CellInstance
import org.platanios.tensorflow.api.learn.layers.{Layer, LayerInstance}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple
import org.platanios.tensorflow.api.ops.seq2seq.decoders.{BasicDecoder, BeamSearchDecoder, GooglePenalty}

/**
  * @author Emmanouil Antonios Platanios
  */
trait Decoder[S, SS] {
  val name: String = "Decoder"

  def trainLayer: Layer[((Output, Output, Output), Tuple[Output, Seq[S]]), (Output, Output)]
  def inferLayer: Layer[((Output, Output), Tuple[Output, Seq[S]]), (Output, Output)]
}

class GNMTDecoder[S, SS](
    val configuration: Configuration[S, SS],
    override val name: String = "GNMTDecoder"
)(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) extends Decoder[S, SS] {
  def timeMajor: Boolean = configuration.dataTimeMajor

  override def trainLayer: Layer[((Output, Output, Output), Tuple[Output, Seq[S]]), (Output, Output)] = {
    new Layer[((Output, Output, Output), Tuple[Output, Seq[S]]), (Output, Output)](name) {
      override val layerType: String = "GNMTTrainDecoder"

      override def forward(
          input: ((Output, Output, Output), Tuple[Output, Seq[S]]),
          mode: Mode
      ): LayerInstance[((Output, Output, Output), Tuple[Output, Seq[S]]), (Output, Output)] = {
        val dataType = configuration.dataType
        tf.createWithNameScope(uniquifiedName) {
          // Keep track of trainable and non-trainable variables
          var trainableVariables = Set.empty[Variable]
          var nonTrainableVariables = Set.empty[Variable]

          // Embeddings
          val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
          val embeddings = variable(
            "Embeddings", dataType, Shape(configuration.tgtVocabSize, configuration.numUnits), embeddingsInitializer)
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

  override def inferLayer: Layer[((Output, Output), Tuple[Output, Seq[S]]), (Output, Output)] = {
    new Layer[((Output, Output), Tuple[Output, Seq[S]]), (Output, Output)](name) {
      override val layerType: String = "GNMTDecoder"

      override def forward(
          input: ((Output, Output), Tuple[Output, Seq[S]]),
          mode: Mode
      ): LayerInstance[((Output, Output), Tuple[Output, Seq[S]]), (Output, Output)] = {
        val dataType = configuration.dataType
        tf.createWithNameScope(uniquifiedName) {
          configuration.srcVocab()
          // Keep track of trainable and non-trainable variables
          var trainableVariables = Set.empty[Variable]
          var nonTrainableVariables = Set.empty[Variable]

          // Embeddings
          val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
          val embeddings = variable(
            "Embeddings", dataType, Shape(configuration.tgtVocabSize, configuration.numUnits), embeddingsInitializer)
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
    val dataType = configuration.dataType
    configuration.decoderAttention match {
      case None =>
        val cellInstance = GNMTModel.multiCell(
          configuration.cell, configuration.numUnits, dataType, configuration.numUniLayers,
          configuration.numUniResLayers, configuration.dropout, configuration.residualFn, 0,
          configuration.numGPUs, configuration.randomSeed, s"$name/MultiCell").createCell(mode, embeddedInput.shape)
        (decode(
          inputSequenceLengths, inputState, embeddings, embeddedInput, cellInstance, variableFn, isTrain, mode),
            cellInstance.trainableVariables, cellInstance.nonTrainableVariables)
      case Some(a) =>
        // Ensure memory is batch-major
        val memory = if (timeMajor) encoderOutput.transpose(Tensor(1, 0, 2)) else encoderOutput
        val memoryWeights = variableFn("MemoryWeights", dataType, Shape(memory.shape(-1), configuration.numUnits), null)
        val attention = a.create(memory, memoryWeights.value, inputSequenceLengths, "Attention")
        val attentionWeights = variableFn(
          "AttentionWeights", attention.dataType,
          Shape(configuration.numUnits + memory.shape(-1), configuration.numUnits), null)
        configuration.decoderAttentionArchitecture match {
          case StandardAttention =>
            val cell = GNMTModel.multiCell(
              configuration.cell, configuration.numUnits, dataType, configuration.numUniLayers,
              configuration.numUniResLayers, configuration.dropout, configuration.residualFn, 0,
              configuration.numGPUs, configuration.randomSeed, s"$name/MultiCell")
            val cellInstance = cell.createCell(mode, embeddedInput.shape)
            val attentionCell = tf.AttentionWrapperCell(
              cellInstance.cell, Seq(attention), Seq(attentionWeights.value),
              outputAttention = configuration.decoderOutputAttention)
            val attentionCellInstance = CellInstance(
              cell = attentionCell, trainableVariables = cellInstance.trainableVariables + attentionWeights,
              nonTrainableVariables = cellInstance.nonTrainableVariables)
            (decode(
              inputSequenceLengths, attentionCell.initialState(inputState, dataType), embeddings, embeddedInput,
              attentionCellInstance, variableFn, isTrain, mode),
                cellInstance.trainableVariables + attentionWeights,
                cellInstance.nonTrainableVariables)
          case GNMTAttention(useNewAttention) =>
            val cells = GNMTModel.cells(
              configuration.cell, configuration.numUnits, dataType, configuration.numUniLayers,
              configuration.numUniResLayers, configuration.dropout, configuration.residualFn, 0,
              configuration.numGPUs, configuration.randomSeed, s"$name/MultiCell")
            ???
        }
    }
  }

  protected def decode[DS, DSS](
      inputSequenceLengths: Output,
      inputState: DS,
      embeddings: Variable,
      embeddedInput: Output,
      cellInstance: CellInstance[Output, Shape, DS, DSS],
      variableFn: (String, DataType, Shape, tf.VariableInitializer) => Variable,
      isTrain: Boolean,
      mode: Mode
  )(implicit
      evS: WhileLoopVariable.Aux[DS, DSS]
  ): (Output, Output) = {
    val outputWeights = variableFn(
      "OutWeights", configuration.dataType, Shape(cellInstance.cell.outputShape(-1), configuration.tgtVocabSize),
      tf.RandomUniformInitializer(-0.1f, 0.1f))
    val outputLayer = (logits: Output) => tf.linear(logits, outputWeights.value)
    if (isTrain) {
      val helper = BasicDecoder.TrainingHelper(embeddedInput, inputSequenceLengths, configuration.dataTimeMajor)
      val decoder = BasicDecoder(cellInstance.cell, inputState, helper, outputLayer)
      val tuple = decoder.decode(
        outputTimeMajor = configuration.dataTimeMajor, parallelIterations = configuration.parallelIterations,
        swapMemory = configuration.swapMemory)
      val lengths = tuple._3
      mode match {
        case INFERENCE | EVALUATION => (tuple._1.sample, lengths)
        case _ => (tuple._1.rnnOutput, lengths)
      }
    } else {
      val embeddingFn = (o: Output) => tf.embeddingLookup(embeddings, o)
      val tgtBosID = configuration.tgtVocab().lookup(tf.constant(configuration.beginOfSequenceToken)).cast(INT32)
      val tgtEosID = configuration.tgtVocab().lookup(tf.constant(configuration.endOfSequenceToken)).cast(INT32)
      if (configuration.inferBeamWidth > 1) {
        val decoder = BeamSearchDecoder(
          cellInstance.cell, inputState,
          embeddingFn, tf.fill(INT32, tf.shape(inputSequenceLengths))(tgtBosID), tgtEosID, configuration.inferBeamWidth,
          GooglePenalty(configuration.inferLengthPenaltyWeight), outputLayer)
        val tuple = decoder.decode(
          outputTimeMajor = configuration.dataTimeMajor,
          maximumIterations = inferMaxLength(tf.max(inputSequenceLengths)),
          parallelIterations = configuration.parallelIterations, swapMemory = configuration.swapMemory)
        (tuple._1.predictedIDs(---, 0), tuple._3(---, 0).cast(INT32))
      } else {
        val decHelper = BasicDecoder.GreedyEmbeddingHelper[DS](
          embeddingFn, tf.fill(INT32, tf.shape(inputSequenceLengths))(tgtBosID), tgtEosID)
        val decoder = BasicDecoder(cellInstance.cell, inputState, decHelper, outputLayer)
        val tuple = decoder.decode(
          outputTimeMajor = configuration.dataTimeMajor,
          maximumIterations = inferMaxLength(tf.max(inputSequenceLengths)),
          parallelIterations = configuration.parallelIterations, swapMemory = configuration.swapMemory)
        (tuple._1.sample, tuple._3)
      }
    }
  }

  /** Returns the maximum sequence length to consider while decoding during inference, given the provided source
    * sequence length. */
  protected def inferMaxLength(srcLength: Output): Output = {
    if (configuration.dataTgtMaxLength != -1)
      tf.constant(configuration.dataTgtMaxLength)
    else
      tf.round(tf.max(srcLength) * configuration.decoderMaxLengthFactor).cast(INT32)
  }
}

object GNMTDecoder {
  def apply[S, SS](configuration: Configuration[S, SS], name: String = "GNMTDecoder")(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): GNMTDecoder[S, SS] = {
    new GNMTDecoder(configuration, name)(evS, evSDropout)
  }
}
