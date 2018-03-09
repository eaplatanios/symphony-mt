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
import org.platanios.symphony.mt.models.{DeviceManager, ParameterManager, RNNModel, Stage}
import org.platanios.symphony.mt.models.rnn.attention.RNNAttention
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.Output
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
    val outputAttention: Boolean = true
)(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S],
    evAS: WhileLoopVariable.Aux[AS, ASS]
) extends RNNDecoder[S, SS]()(evS, evSDropout) {
  override def create(
      config: RNNModel.Config[_, _],
      srcLanguage: Output,
      tgtLanguage: Output,
      encoderState: (Tuple[Output, Seq[S]], Output, Output),
      beginOfSequenceToken: String,
      endOfSequenceToken: String,
      tgtSequences: Output = null,
      tgtSequenceLengths: Output = null
  )(
      mode: Mode,
      env: Environment,
      parameterManager: ParameterManager,
      deviceManager: DeviceManager
  )(implicit
      stage: Stage
  ): RNNDecoder.Output = {
    // Embeddings
    val embeddings = parameterManager.wordEmbeddings(tgtLanguage)

    // RNN cell
    val numResLayers = if (residual && numLayers > 1) numLayers - 1 else 0
    val uniCell = attention match {
      case None =>
        RNNModel.multiCell(
          cell, numUnits, numUnits, dataType, numLayers, numResLayers, dropout, residualFn, config.env.randomSeed,
          "MultiUniCell")(mode, env, parameterManager, deviceManager)
      case Some(_) =>
        RNNModel.multiCell(
          cell, 2 * numUnits, numUnits, dataType, numLayers, numResLayers, dropout, residualFn, config.env.randomSeed,
          "MultiUniCell")(mode, env, parameterManager, deviceManager)
    }

    // Use attention if necessary and create the decoder RNN
    var initialState = encoderState._1.state
    var memory = if (config.timeMajor) encoderState._1.output.transpose(Tensor(1, 0, 2)) else encoderState._1.output
    var memorySequenceLengths = encoderState._2
    if (config.beamWidth > 1 && !mode.isTraining) {
      // TODO: Find a way to remove the need for this tiling that is external to the beam search decoder.
      initialState = BeamSearchDecoder.tileForBeamSearch(initialState, config.beamWidth)
      memory = BeamSearchDecoder.tileForBeamSearch(memory, config.beamWidth)
      memorySequenceLengths = BeamSearchDecoder.tileForBeamSearch(memorySequenceLengths, config.beamWidth)
    }

    attention match {
      case None =>
        decode(
          config, encoderState._2, tgtLanguage, tgtSequences, tgtSequenceLengths, initialState,
          embeddings, uniCell, encoderState._3, beginOfSequenceToken, endOfSequenceToken)(mode, parameterManager)
      case Some(attentionCreator) =>
        val (attentionCell, attentionInitialState) = attentionCreator.create(
          srcLanguage, tgtLanguage, uniCell, memory, memorySequenceLengths,
          numUnits, numUnits, initialState, useAttentionLayer = true,
          outputAttention = outputAttention)(mode, parameterManager)
        decode(
          config, encoderState._2, tgtLanguage, tgtSequences, tgtSequenceLengths,
          attentionInitialState, embeddings, attentionCell, encoderState._3, beginOfSequenceToken,
          endOfSequenceToken)(mode, parameterManager)
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
      outputAttention: Boolean = false
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S],
      evAS: WhileLoopVariable.Aux[AS, ASS]
  ): UnidirectionalRNNDecoder[S, SS, AS, ASS] = {
    new UnidirectionalRNNDecoder[S, SS, AS, ASS](
      cell, numUnits, numLayers, dataType, residual, dropout, residualFn, attention,
      outputAttention)(evS, evSDropout, evAS)
  }
}
