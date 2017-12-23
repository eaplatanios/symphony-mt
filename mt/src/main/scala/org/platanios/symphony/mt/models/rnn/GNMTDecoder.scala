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

import org.platanios.symphony.mt.data.{DataConfig, Vocabulary}
import org.platanios.symphony.mt.models.InferConfig
import org.platanios.symphony.mt.models.attention.{Attention, LuongAttention}
import org.platanios.symphony.mt.{Environment, Language}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.attention.{AttentionWrapperCell, AttentionWrapperState}
import org.platanios.tensorflow.api.ops.rnn.cell.{RNNCell, Tuple}
import org.platanios.tensorflow.api.ops.seq2seq.decoders.{BasicDecoder, BeamSearchDecoder, GooglePenalty}

/**
  * @author Emmanouil Antonios Platanios
  */
class GNMTDecoder[S, SS](
    val tgtLanguage: Language,
    val tgtVocabulary: Vocabulary,
    val env: Environment,
    val rnnConfig: RNNConfig,
    val dataConfig: DataConfig,
    val inferConfig: InferConfig,
    val cell: Cell[S, SS],
    val numUnits: Int,
    val numLayers: Int,
    val numResLayers: Int,
    val dataType: DataType = FLOAT32,
    val dropout: Option[Float] = None,
    val attention: Attention = LuongAttention(scaled = true),
    val useNewAttention: Boolean = false
)(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) extends RNNDecoder[S, SS]()(evS, evSDropout) {
  override def create(
      encoderTuple: Tuple[Output, Seq[S]], inputSequenceLengths: Output,
      targetSequences: Output, targetSequenceLengths: Output, mode: Mode
  ): RNNDecoder.Output = {
    // Embeddings
    val embeddings = Model.embeddings(dataType, tgtVocabulary.size, numUnits, "Embeddings")

    // RNN cells
    val cells = Model.cells(
      cell, numUnits, dataType, numLayers, numResLayers, dropout,
      Some(GNMTDecoder.residualFn[Output, Shape]), 0, env.numGPUs, env.randomSeed, "Cells")

    // Attention
    val bottomCell = cells.head
    var initialState = encoderTuple.state
    var memory = {
      if (rnnConfig.timeMajor)
        encoderTuple.output.transpose(Tensor(1, 0, 2))
      else
        encoderTuple.output
    }
    var memorySequenceLengths = inputSequenceLengths
    if (inferConfig.beamWidth > 1 && !mode.isTraining) {
      // TODO: Find a way to remove the need for this tiling that is external to the beam search decoder.
      initialState = BeamSearchDecoder.tileForBeamSearch(initialState, inferConfig.beamWidth)
      memory = BeamSearchDecoder.tileForBeamSearch(memory, inferConfig.beamWidth)
      memorySequenceLengths = BeamSearchDecoder.tileForBeamSearch(memorySequenceLengths, inferConfig.beamWidth)
    }
    val (attentionCell, attentionInitialState) = attention.create[S, SS](
      bottomCell, memory, memorySequenceLengths, numUnits, numUnits, initialState.head, useAttentionLayer = false,
      outputAttention = false, mode)
    val multiCell = GNMTDecoder.MultiCell[S, SS](
      attentionCell, cells.tail.map(_.createCell(mode, Shape(2 * numUnits))), useNewAttention)
    decode(
      inputSequenceLengths, targetSequences, targetSequenceLengths, (attentionInitialState, initialState.tail),
      embeddings, multiCell, mode)
  }

  protected def decode[DS, DSS](
      inputSequenceLengths: Output,
      targetSequences: Output,
      targetSequenceLengths: Output,
      initialState: DS,
      embeddings: Variable,
      cell: RNNCell[Output, Shape, DS, DSS],
      mode: Mode
  )(implicit
      evS: WhileLoopVariable.Aux[DS, DSS]
  ): RNNDecoder.Output = {
    val outputWeights = tf.variable(
      "OutWeights", embeddings.dataType, Shape(cell.outputShape(-1), tgtVocabulary.size),
      tf.RandomUniformInitializer(-0.1f, 0.1f))
    val outputLayer = (logits: Output) => tf.linear(logits, outputWeights.value)
    if (mode.isTraining) {
      // Time-major transpose
      val transposedSequences = if (rnnConfig.timeMajor) targetSequences.transpose() else targetSequences
      val embeddedSequences = tf.embeddingLookup(embeddings, transposedSequences)

      // Decoder RNN
      val helper = BasicDecoder.TrainingHelper(embeddedSequences, targetSequenceLengths, rnnConfig.timeMajor)
      val decoder = BasicDecoder(cell, initialState, helper, outputLayer)
      val tuple = decoder.decode(
        outputTimeMajor = rnnConfig.timeMajor, parallelIterations = rnnConfig.parallelIterations,
        swapMemory = rnnConfig.swapMemory)
      RNNDecoder.Output(tuple._1.rnnOutput, tuple._3)
    } else {
      // Decoder embeddings
      val embeddingFn = (o: Output) => tf.embeddingLookup(embeddings, o)
      val tgtVocabLookupTable = tgtVocabulary.lookupTable()
      val tgtBosID = tgtVocabLookupTable.lookup(tf.constant(dataConfig.beginOfSequenceToken)).cast(INT32)
      val tgtEosID = tgtVocabLookupTable.lookup(tf.constant(dataConfig.endOfSequenceToken)).cast(INT32)
      val maxDecodingLength = {
        if (dataConfig.tgtMaxLength != -1)
          tf.constant(dataConfig.tgtMaxLength)
        else
          tf.round(tf.max(tf.max(inputSequenceLengths)) * inferConfig.decoderMaxLengthFactor).cast(INT32)
      }

      // Decoder RNN
      if (inferConfig.beamWidth > 1) {
        val decoder = BeamSearchDecoder(
          cell, initialState, embeddingFn, tf.fill(INT32, tf.shape(inputSequenceLengths)(0).expandDims(0))(tgtBosID),
          tgtEosID, inferConfig.beamWidth, GooglePenalty(inferConfig.lengthPenaltyWeight), outputLayer)
        val tuple = decoder.decode(
          outputTimeMajor = rnnConfig.timeMajor, maximumIterations = maxDecodingLength,
          parallelIterations = rnnConfig.parallelIterations, swapMemory = rnnConfig.swapMemory)
        RNNDecoder.Output(tuple._1.predictedIDs(---, 0), tuple._3(---, 0).cast(INT32))
      } else {
        val decHelper = BasicDecoder.GreedyEmbeddingHelper[DS](
          embeddingFn, tf.fill(INT32, tf.shape(inputSequenceLengths)(0).expandDims(0))(tgtBosID), tgtEosID)
        val decoder = BasicDecoder(cell, initialState, decHelper, outputLayer)
        val tuple = decoder.decode(
          outputTimeMajor = rnnConfig.timeMajor, maximumIterations = maxDecodingLength,
          parallelIterations = rnnConfig.parallelIterations, swapMemory = rnnConfig.swapMemory)
        RNNDecoder.Output(tuple._1.sample, tuple._3)
      }
    }
  }
}

object GNMTDecoder {
  def apply[S, SS](
      tgtLanguage: Language,
      tgtVocabulary: Vocabulary,
      env: Environment,
      rnnConfig: RNNConfig,
      dataConfig: DataConfig,
      inferConfig: InferConfig,
      cell: Cell[S, SS],
      numUnits: Int,
      numLayers: Int,
      numResLayers: Int,
      dataType: DataType = FLOAT32,
      dropout: Option[Float] = None,
      attention: Attention = LuongAttention(scaled = true),
      useNewAttention: Boolean = false
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): GNMTDecoder[S, SS] = {
    new GNMTDecoder[S, SS](
      tgtLanguage, tgtVocabulary, env, rnnConfig, dataConfig, inferConfig, cell, numUnits, numLayers, numResLayers,
      dataType, dropout, attention, useNewAttention)(evS, evSDropout)
  }

  /** GNMT model residual function that handles inputs and outputs with different sizes (due to attention). */
  private[GNMTDecoder] def residualFn[O, OS](input: O, output: O)(implicit evO: WhileLoopVariable.Aux[O, OS]): O = {
    evO.fromOutputs(output, evO.outputs(input).zip(evO.outputs(output)).map {
      case (i, o) =>
        val oLastDim = tf.shape(o)(-1)
        val iLastDim = tf.shape(i)(-1)
        val actualInput = tf.split(i, tf.stack(Seq(oLastDim, iLastDim - oLastDim)), axis = -1).head
        actualInput.shape.assertIsCompatibleWith(o.shape)
        actualInput + o
    })
  }

  /** GNMT RNN cell that is composed by applying an attention cell and then a sequence of RNN cells in order, all being
    * fed the same attention as input.
    *
    * This means that the output of each RNN is fed to the next one as input, while the states remain separate.
    * Furthermore, the attention layer is used first as the bottom layer and then the same attention is fed to all upper
    * layers. That is either the previous attention, or the new attention generated by the bottom layer (if
    * `useNewAttention` is set to `true`).
    *
    * Note that this class does no variable management at all. Variable sharing should be handled based on the RNN cells
    * the caller provides to this class. The learn API provides a layer version of this class that also does some
    * management of the variables involved.
    *
    * @param  attentionCell   Attention cell to use.
    * @param  cells           Cells being stacked together.
    * @param  useNewAttention Boolean value specifying whether to use the attention generated from the current step
    *                         bottom layer's output as input to all upper layers, or the previous attention (i.e., same
    *                         as the one that's input to the bottom layer).
    * @param  name            Name prefix used for all new ops.
    *
    * @author Emmanouil Antonios Platanios
    */
  class MultiCell[S, SS](
      val attentionCell: AttentionWrapperCell[S, SS],
      val cells: Seq[RNNCell[Output, Shape, S, SS]],
      val useNewAttention: Boolean = false,
      val name: String = "GNMTMultiCell"
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS]
  ) extends RNNCell[
      Output, Shape, (AttentionWrapperState[S, SS], Seq[S]), ((SS, Shape, Shape, Seq[Shape], Seq[Shape]), Seq[SS])] {
    override def outputShape: Shape = cells.last.outputShape

    override def stateShape: ((SS, Shape, Shape, Seq[Shape], Seq[Shape]), Seq[SS]) = {
      (attentionCell.stateShape, cells.map(_.stateShape))
    }

    override def forward(
        input: Tuple[Output, (AttentionWrapperState[S, SS], Seq[S])]
    ): Tuple[Output, (AttentionWrapperState[S, SS], Seq[S])] = tf.createWithNameScope(name) {
      val minusOne = tf.constant(-1)
      val nextAttentionTuple = attentionCell(Tuple(input.output, input.state._1))
      var currentInput = nextAttentionTuple.output
      val state = cells.zip(input.state._2).map {
        case (cell, s) =>
          val concatenatedInput = {
            if (useNewAttention)
              tf.concatenate(Seq(currentInput, nextAttentionTuple.state.attention), axis = minusOne)
            else
              tf.concatenate(Seq(currentInput, input.state._1.attention), axis = minusOne)
          }
          val nextTuple = cell(Tuple(concatenatedInput, s))
          currentInput = nextTuple.output
          nextTuple.state
      }
      Tuple(currentInput, (nextAttentionTuple.state, state))
    }
  }

  object MultiCell {
    def apply[S, SS](
        attentionCell: AttentionWrapperCell[S, SS],
        cells: Seq[RNNCell[Output, Shape, S, SS]],
        useNewAttention: Boolean = false,
        name: String = "GNMTMultiCell"
    )(implicit
        evS: WhileLoopVariable.Aux[S, SS]
    ): MultiCell[S, SS] = {
      new MultiCell(attentionCell, cells, useNewAttention, name)(evS)
    }
  }
}
