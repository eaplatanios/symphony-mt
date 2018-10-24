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
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.rnn.attention.{AttentionWrapperCell, AttentionWrapperState}
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple

/**
  * @author Emmanouil Antonios Platanios
  */
class GNMTDecoder[T: TF : IsNotQuantized, State: OutputStructure, AttentionState: OutputStructure, StateShape, AttentionStateShape](
    val cell: Cell[T, State, StateShape],
    val numUnits: Int,
    val numLayers: Int,
    val numResLayers: Int,
    val attention: RNNAttention[T, AttentionState, AttentionStateShape],
    val dropout: Option[Float] = None,
    val useNewAttention: Boolean = true
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

    // RNN cells
    val cells = (0 until numLayers).foldLeft(Seq.empty[tf.RNNCell[Output[T], State, Shape, StateShape]])((cells, i) => {
      val cellNumInputs = if (i == 0) 2 * numUnits else cells(i - 1).outputShape.apply(-1) + numUnits
      cells :+ RNNModel.cell[T, State, StateShape](
        cell = cell,
        numInputs = cellNumInputs,
        numUnits = numUnits,
        dropout = dropout,
        residualFn = {
          if (i >= numLayers - numResLayers) {
            Some((i: Output[T], o: Output[T]) => {
              // This residual function can handle inputs and outputs of different sizes (due to attention).
              val oLastDim = tf.shape(o).slice(-1)
              val iLastDim = tf.shape(i).slice(-1)
              val actualInput = tf.split(i, tf.stack(Seq(oLastDim, iLastDim - oLastDim)), axis = -1).head
              actualInput.shape.assertIsCompatibleWith(o.shape)
              actualInput + o
            })
          } else {
            None
          }
        },
        device = deviceManager.nextDevice(env),
        seed = config.env.randomSeed,
        name = s"Cell$i")
    })

    // Attention
    var initialState = encoderState._1.state
    var memory = if (config.timeMajor) encoderState._1.output.transpose(Tensor(1, 0, 2)) else encoderState._1.output
    var memorySequenceLengths = encoderState._2
    if (config.beamWidth > 1 && !mode.isTraining) {
      // TODO: Find a way to remove the need for this tiling that is external to the beam search decoder.
      initialState = Decoder.tileForBeamSearch(initialState, config.beamWidth)
      memory = Decoder.tileForBeamSearch(memory, config.beamWidth)
      memorySequenceLengths = Decoder.tileForBeamSearch(memorySequenceLengths, config.beamWidth)
    }
    val (attentionCell, attentionInitialState) = attention.create[State, StateShape](
      cells.head, memory, memorySequenceLengths, numUnits,
      numUnits, initialState.head,
      useAttentionLayer = false,
      outputAttention = false)
    val multiCell = GNMTDecoder.StackedCell[T, State, AttentionState, StateShape, AttentionStateShape](
      attentionCell, cells.tail, useNewAttention)

    decode(
      decodingMode, config, encoderState._2, tgtSequences, tgtSequenceLengths,
      (attentionInitialState, initialState.tail), embeddings(_).castTo[T], multiCell,
      encoderState._3, beginOfSequenceToken, endOfSequenceToken)
  }
}

object GNMTDecoder {
  def apply[T: TF : IsNotQuantized, State: OutputStructure, AttentionState: OutputStructure, StateShape, AttentionStateShape](
      cell: Cell[T, State, StateShape],
      numUnits: Int,
      numLayers: Int,
      numResLayers: Int,
      attention: RNNAttention[T, AttentionState, AttentionStateShape],
      dropout: Option[Float] = None,
      useNewAttention: Boolean = true
  )(implicit
      evOutputToShapeState: OutputToShape.Aux[State, StateShape],
      evOutputToShapeAttentionState: OutputToShape.Aux[AttentionState, AttentionStateShape]
  ): GNMTDecoder[T, State, AttentionState, StateShape, AttentionStateShape] = {
    new GNMTDecoder[T, State, AttentionState, StateShape, AttentionStateShape](
      cell, numUnits, numLayers, numResLayers, attention, dropout, useNewAttention)
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
  class StackedCell[T: TF : IsNotQuantized, State: OutputStructure, AttentionState: OutputStructure, StateShape, AttentionStateShape](
      val attentionCell: AttentionWrapperCell[T, State, AttentionState, StateShape, AttentionStateShape],
      val cells: Seq[tf.RNNCell[Output[T], State, Shape, StateShape]],
      val useNewAttention: Boolean = false,
      val name: String = "GNMTMultiCell"
  )(implicit
      evOutputToShapeState: OutputToShape.Aux[State, StateShape],
      evOutputToShapeAttentionState: OutputToShape.Aux[AttentionState, AttentionStateShape]
  ) extends tf.RNNCell[Output[T], (AttentionWrapperState[T, State, Seq[AttentionState]], Seq[State]), Shape, ((StateShape, Shape, Shape, Seq[Shape], Seq[Shape], Seq[AttentionStateShape]), Seq[StateShape])] {
    override def outputShape: Shape = {
      cells.last.outputShape
    }

    override def stateShape: ((StateShape, Shape, Shape, Seq[Shape], Seq[Shape], Seq[AttentionStateShape]), Seq[StateShape]) = {
      (attentionCell.stateShape, cells.map(_.stateShape))
    }

    override def forward(
        input: Tuple[Output[T], (AttentionWrapperState[T, State, Seq[AttentionState]], Seq[State])]
    ): Tuple[Output[T], (AttentionWrapperState[T, State, Seq[AttentionState]], Seq[State])] = {
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

  object StackedCell {
    def apply[T: TF : IsNotQuantized, State: OutputStructure, AttentionState: OutputStructure, StateShape, AttentionStateShape](
        attentionCell: AttentionWrapperCell[T, State, AttentionState, StateShape, AttentionStateShape],
        cells: Seq[tf.RNNCell[Output[T], State, Shape, StateShape]],
        useNewAttention: Boolean = false,
        name: String = "GNMTMultiCell"
    )(implicit
        evOutputToShapeState: OutputToShape.Aux[State, StateShape],
        evOutputToShapeAttentionState: OutputToShape.Aux[AttentionState, AttentionStateShape]
    ): StackedCell[T, State, AttentionState, StateShape, AttentionStateShape] = {
      new StackedCell(attentionCell, cells, useNewAttention, name)
    }
  }
}
