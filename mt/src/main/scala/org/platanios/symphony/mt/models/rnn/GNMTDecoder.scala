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
import org.platanios.tensorflow.api.ops.rnn.attention.{AttentionWrapperCell, AttentionWrapperState}
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple
import org.platanios.tensorflow.api.ops.seq2seq.decoders.BeamSearchDecoder

/**
  * @author Emmanouil Antonios Platanios
  */
class GNMTDecoder[S, SS, AS, ASS](
    val cell: Cell[S, SS],
    val numUnits: Int,
    val numLayers: Int,
    val numResLayers: Int,
    val attention: RNNAttention[AS, ASS],
    val dataType: DataType = FLOAT32,
    val dropout: Option[Float] = None,
    val useNewAttention: Boolean = true
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
  )(implicit
      stage: Stage,
      mode: Mode,
      env: Environment,
      parameterManager: ParameterManager,
      deviceManager: DeviceManager
  ): RNNDecoder.Output = {
    // Embeddings
    val embeddings = parameterManager.wordEmbeddings(tgtLanguage)

    // RNN cells
    val cells = (0 until numLayers).foldLeft(Seq.empty[tf.RNNCell[Output, Shape, S, SS]])((cells, i) => {
      val cellNumInputs = if (i == 0) 2 * numUnits else cells(i - 1).outputShape(-1) + numUnits
      cells :+ RNNModel.cell(
        cell, cellNumInputs, numUnits, dataType, dropout,
        if (i >= numLayers - numResLayers) Some(GNMTDecoder.residualFn[Output, Shape]) else None,
        deviceManager.nextDevice(env), config.env.randomSeed, s"Cell$i")(mode, parameterManager)
    })

    // Attention
    var initialState = encoderState._1.state
    var memory = if (config.timeMajor) encoderState._1.output.transpose(Tensor(1, 0, 2)) else encoderState._1.output
    var memorySequenceLengths = encoderState._2
    if (config.beamWidth > 1 && !mode.isTraining) {
      // TODO: Find a way to remove the need for this tiling that is external to the beam search decoder.
      initialState = BeamSearchDecoder.tileForBeamSearch(initialState, config.beamWidth)
      memory = BeamSearchDecoder.tileForBeamSearch(memory, config.beamWidth)
      memorySequenceLengths = BeamSearchDecoder.tileForBeamSearch(memorySequenceLengths, config.beamWidth)
    }
    val (attentionCell, attentionInitialState) = attention.create[S, SS](
      srcLanguage, tgtLanguage, cells.head, memory, memorySequenceLengths, numUnits,
      numUnits, initialState.head, useAttentionLayer = false, outputAttention = false)
    val multiCell = GNMTDecoder.MultiCell[S, SS, AS, ASS](attentionCell, cells.tail, useNewAttention)
    decode(
      config, encoderState._2, tgtLanguage, tgtSequences, tgtSequenceLengths,
      (attentionInitialState, initialState.tail), embeddings, multiCell, encoderState._3,
      beginOfSequenceToken, endOfSequenceToken)
  }
}

object GNMTDecoder {
  def apply[S, SS, AS, ASS](
      cell: Cell[S, SS],
      numUnits: Int,
      numLayers: Int,
      numResLayers: Int,
      attention: RNNAttention[AS, ASS],
      dataType: DataType = FLOAT32,
      dropout: Option[Float] = None,
      useNewAttention: Boolean = true
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S],
      evAS: WhileLoopVariable.Aux[AS, ASS]
  ): GNMTDecoder[S, SS, AS, ASS] = {
    new GNMTDecoder[S, SS, AS, ASS](
      cell, numUnits, numLayers, numResLayers, attention, dataType, dropout, useNewAttention)(evS, evSDropout, evAS)
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
  class MultiCell[S, SS, AS, ASS](
      val attentionCell: AttentionWrapperCell[S, SS, AS, ASS],
      val cells: Seq[tf.RNNCell[Output, Shape, S, SS]],
      val useNewAttention: Boolean = false,
      val name: String = "GNMTMultiCell"
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evAS: WhileLoopVariable.Aux[AS, ASS]
  ) extends tf.RNNCell[
      Output, Shape,
      (AttentionWrapperState[S, SS, Seq[AS], Seq[ASS]], Seq[S]),
      ((SS, Shape, Shape, Seq[Shape], Seq[Shape], Seq[ASS]), Seq[SS])] {
    override def outputShape: Shape = cells.last.outputShape

    override def stateShape: ((SS, Shape, Shape, Seq[Shape], Seq[Shape], Seq[ASS]), Seq[SS]) = {
      (attentionCell.stateShape, cells.map(_.stateShape))
    }

    override def forward(
        input: Tuple[Output, (AttentionWrapperState[S, SS, Seq[AS], Seq[ASS]], Seq[S])]
    ): Tuple[Output, (AttentionWrapperState[S, SS, Seq[AS], Seq[ASS]], Seq[S])] = tf.createWithNameScope(name) {
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
    def apply[S, SS, AS, ASS](
        attentionCell: AttentionWrapperCell[S, SS, AS, ASS],
        cells: Seq[tf.RNNCell[Output, Shape, S, SS]],
        useNewAttention: Boolean = false,
        name: String = "GNMTMultiCell"
    )(implicit
        evS: WhileLoopVariable.Aux[S, SS],
        evAS: WhileLoopVariable.Aux[AS, ASS]
    ): MultiCell[S, SS, AS, ASS] = {
      new MultiCell(attentionCell, cells, useNewAttention, name)(evS, evAS)
    }
  }
}
