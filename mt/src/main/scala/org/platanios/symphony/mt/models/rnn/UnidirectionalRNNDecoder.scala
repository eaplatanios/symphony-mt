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

package org.platanios.symphony.mt.models.rnn

import org.platanios.symphony.mt.Environment
import org.platanios.symphony.mt.models.StateBasedModel
import org.platanios.symphony.mt.models.rnn.attention.RNNAttention
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple
import org.platanios.tensorflow.api.ops.seq2seq.decoders.BeamSearchDecoder

/**
  * @author Emmanouil Antonios Platanios
  */
class UnidirectionalRNNDecoder[S, SS, AS, ASS](
    val cell: Cell[S, SS],
    val numUnits: Int,
    val numLayers: Int,
    val dataType: DataType = FLOAT32,
    val residual: Boolean = false,
    val dropout: Option[Float] = None,
    val residualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output),
    val attention: Option[RNNAttention[AS, ASS]] = None,
    val outputAttention: Boolean = true,
    override val timeMajor: Boolean = false,
    // Inference
    override val beamWidth: Int = 10,
    override val lengthPenaltyWeight: Float = 0.0f,
    override val decoderMaxLengthFactor: Float = 2.0f
)(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S],
    evAS: WhileLoopVariable.Aux[AS, ASS]
) extends RNNDecoder[S, SS](timeMajor, beamWidth, lengthPenaltyWeight, decoderMaxLengthFactor)(evS, evSDropout) {
  override def create(
      env: Environment,
      encoderTuple: Tuple[Output, Seq[S]],
      srcSequenceLengths: Output,
      tgtVocab: Vocabulary,
      tgtMaxLength: Int,
      beginOfSequenceToken: String,
      endOfSequenceToken: String,
      tgtSequences: Output = null,
      tgtSequenceLengths: Output = null,
      mode: Mode
  ): RNNDecoder.Output = {
    // Embeddings
    val embeddings = StateBasedModel.embeddings(dataType, tgtVocab.size, numUnits, "Embeddings")

    // RNN cell
    val numResLayers = if (residual && numLayers > 1) numLayers - 1 else 0
    val uniCell = StateBasedModel.multiCell(
      cell, numUnits, dataType, numLayers, numResLayers, dropout,
      residualFn, 0, env.numGPUs, env.firstGPU, env.randomSeed, "MultiUniCell")

    // Use attention if necessary and create the decoder RNN
    var initialState = encoderTuple.state
    var memory = if (timeMajor) encoderTuple.output.transpose(Tensor(1, 0, 2)) else encoderTuple.output
    var memorySequenceLengths = srcSequenceLengths
    if (beamWidth > 1 && !mode.isTraining) {
      // TODO: Find a way to remove the need for this tiling that is external to the beam search decoder.
      initialState = BeamSearchDecoder.tileForBeamSearch(initialState, beamWidth)
      memory = BeamSearchDecoder.tileForBeamSearch(memory, beamWidth)
      memorySequenceLengths = BeamSearchDecoder.tileForBeamSearch(memorySequenceLengths, beamWidth)
    }

    attention match {
      case None =>
        decode(
          env, srcSequenceLengths, tgtSequences, tgtSequenceLengths, initialState,
          embeddings, uniCell.createCell(mode, Shape(numUnits)), tgtVocab, tgtMaxLength,
          beginOfSequenceToken, endOfSequenceToken, mode)
      case Some(attentionCreator) =>
        val (attentionCell, attentionInitialState) = attentionCreator.create(
          uniCell, memory, memorySequenceLengths, numUnits, numUnits, initialState, useAttentionLayer = true,
          outputAttention = outputAttention, mode)
        decode(
          env, srcSequenceLengths, tgtSequences, tgtSequenceLengths, attentionInitialState,
          embeddings, attentionCell, tgtVocab, tgtMaxLength, beginOfSequenceToken, endOfSequenceToken, mode)
    }
  }
}

object UnidirectionalRNNDecoder {
  def apply[S, SS, AS, ASS](
      cell: Cell[S, SS],
      numUnits: Int,
      numLayers: Int,
      dataType: DataType = FLOAT32,
      residual: Boolean = false,
      dropout: Option[Float] = None,
      residualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output),
      attention: Option[RNNAttention[AS, ASS]] = None,
      outputAttention: Boolean = false,
      timeMajor: Boolean = false,
      // Inference
      beamWidth: Int = 10,
      lengthPenaltyWeight: Float = 0.0f,
      decoderMaxLengthFactor: Float = 2.0f
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S],
      evAS: WhileLoopVariable.Aux[AS, ASS]
  ): UnidirectionalRNNDecoder[S, SS, AS, ASS] = {
    new UnidirectionalRNNDecoder[S, SS, AS, ASS](
      cell, numUnits, numLayers, dataType, residual, dropout, residualFn, attention, outputAttention, timeMajor,
      beamWidth, lengthPenaltyWeight, decoderMaxLengthFactor)(evS, evSDropout, evAS)
  }
}
