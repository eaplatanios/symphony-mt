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
import org.platanios.symphony.mt.models.parameters.ParameterManager
import org.platanios.symphony.mt.models._
import org.platanios.symphony.mt.models.rnn.attention.RNNAttention
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape, Zero}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple

/**
  * @author Emmanouil Antonios Platanios
  */
class UnidirectionalRNNDecoder[T: TF : IsNotQuantized, State: OutputStructure, AttentionState: OutputStructure, StateShape, AttentionStateShape](
    val cell: Cell[T, State, StateShape],
    val numUnits: Int,
    val numLayers: Int,
    val residual: Boolean = false,
    val dropout: Option[Float] = None,
    val residualFn: Option[(Output[T], Output[T]) => Output[T]] = None,
    val attention: Option[RNNAttention[T, AttentionState, AttentionStateShape]] = None,
    val outputAttention: Boolean = true
)(implicit
    evOutputToShapeState: OutputToShape.Aux[State, StateShape],
    evOutputToShapeAttentionState: OutputToShape.Aux[AttentionState, AttentionStateShape]
) extends RNNDecoder[T, State]() {
  override def create[O: TF](
      decodingMode: Model.DecodingMode[O],
      config: RNNModel.Config[T, _],
      encoderState: (Tuple[Output[T], Seq[State]], Output[Int], Output[Int]),
      beginOfSequenceToken: String,
      endOfSequenceToken: String,
      tgtSequences: Output[Int] = null,
      tgtSequenceLengths: Output[Int] = null
  )(implicit
      stage: Stage,
      mode: Mode,
      env: Environment,
      parameterManager: ParameterManager,
      deviceManager: DeviceManager,
      context: Output[Int]
  ): RNNDecoder.DecoderOutput[O] = {
    // Embeddings
    val tgtLanguage = context(1)
    val embeddings = parameterManager.wordEmbeddings(tgtLanguage)

    // RNN cell
    val numResLayers = if (residual && numLayers > 1) numLayers - 1 else 0
    val uniCell = attention match {
      case None =>
        RNNModel.stackedCell[T, State, StateShape](
          cell = cell,
          numInputs = numUnits,
          numUnits = numUnits,
          numLayers = numLayers,
          numResidualLayers = numResLayers,
          dropout = dropout,
          residualFn = residualFn,
          seed = config.env.randomSeed,
          name = "MultiUniCell")
      case Some(_) =>
        RNNModel.stackedCell[T, State, StateShape](
          cell, 2 * numUnits, numUnits, numLayers, numResLayers, dropout,
          residualFn, config.env.randomSeed, "MultiUniCell")
    }

    // Use attention, if necessary, and create the decoder RNN.
    var initialState = encoderState._1.state
    var memory = encoderState._1.output
    var memorySequenceLengths = encoderState._2

    // Transpose the memory tensor, if necessary.
    if (config.timeMajor) {
      memory = encoderState._1.output.transpose(Tensor(1, 0, 2))
    }

    // If using beam search we need to tile the initial state and the memory-related tensors.
    if (config.beamWidth > 1 && !mode.isTraining) {
      // TODO: Find a way to remove the need for this tiling that is external to the beam search decoder.
      initialState = Decoder.tileForBeamSearch(initialState, config.beamWidth)
      memory = Decoder.tileForBeamSearch(memory, config.beamWidth)
      memorySequenceLengths = Decoder.tileForBeamSearch(memorySequenceLengths, config.beamWidth)
    }

    attention match {
      case None =>
        decode(
          decodingMode, config, encoderState._2, tgtSequences, tgtSequenceLengths, initialState,
          embeddings(_).castTo[T], uniCell, encoderState._3, beginOfSequenceToken, endOfSequenceToken)
      case Some(attentionCreator) =>
        val (attentionCell, attentionInitialState) = attentionCreator.create(
          uniCell, memory, memorySequenceLengths, numUnits, numUnits, initialState,
          useAttentionLayer = true, outputAttention = outputAttention)
        decode(
          decodingMode, config, encoderState._2, tgtSequences, tgtSequenceLengths,
          attentionInitialState, embeddings(_).castTo[T], attentionCell, encoderState._3, beginOfSequenceToken,
          endOfSequenceToken)
    }
  }
}

object UnidirectionalRNNDecoder {
  def apply[T: TF : IsNotQuantized, State: OutputStructure, AttentionState: OutputStructure, StateShape, AttentionStateShape](
      cell: Cell[T, State, StateShape],
      numUnits: Int,
      numLayers: Int,
      residual: Boolean = false,
      dropout: Option[Float] = None,
      residualFn: Option[(Output[T], Output[T]) => Output[T]] = None,
      attention: Option[RNNAttention[T, AttentionState, AttentionStateShape]] = None,
      outputAttention: Boolean = false
  )(implicit
      evOutputToShapeState: OutputToShape.Aux[State, StateShape],
      evOutputToShapeAttentionState: OutputToShape.Aux[AttentionState, AttentionStateShape]
  ): UnidirectionalRNNDecoder[T, State, AttentionState, StateShape, AttentionStateShape] = {
    new UnidirectionalRNNDecoder[T, State, AttentionState, StateShape, AttentionStateShape](
      cell, numUnits, numLayers, residual, dropout,
      residualFn, attention, outputAttention)
  }
}
