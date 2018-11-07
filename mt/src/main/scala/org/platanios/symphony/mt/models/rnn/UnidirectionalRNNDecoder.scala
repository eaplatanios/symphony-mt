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

import org.platanios.symphony.mt.models._
import org.platanios.symphony.mt.models.rnn.Utilities._
import org.platanios.symphony.mt.models.rnn.attention.RNNAttention
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.rnn.attention.{Attention, AttentionWrapperCell, AttentionWrapperState}
import org.platanios.tensorflow.api.ops.rnn.cell.RNNCell

/**
  * @author Emmanouil Antonios Platanios
  */
class UnidirectionalRNNDecoder[T: TF : IsNotQuantized, State: OutputStructure, StateShape](
    val cell: Cell[T, State, StateShape],
    val numUnits: Int,
    val numLayers: Int,
    val residual: Boolean = false,
    val dropout: Option[Float] = None,
    val residualFn: Option[(Output[T], Output[T]) => Output[T]] = None
)(implicit
    evOutputToShapeState: OutputToShape.Aux[State, StateShape]
) extends RNNDecoder[T, State, Seq[State], Seq[StateShape]] {
  override def embeddings(
      ids: Output[Int]
  )(implicit context: Context): Output[T] = {
    val embeddingsTable = context.parameterManager.wordEmbeddings(context.tgtLanguageID)
    embeddingsTable(ids).castTo[T]
  }

  override protected def cellAndInitialState(
      encodedSequences: EncodedSequences[T, State],
      tgtSequences: Option[Sequences[Int]]
  )(implicit context: Context): (RNNCell[Output[T], Seq[State], Shape, Seq[StateShape]], Seq[State]) = {
    val numResLayers = if (residual && numLayers > 1) numLayers - 1 else 0
    val uniCell = stackedCell[T, State, StateShape](
      cell = cell,
      numInputs = numUnits,
      numUnits = numUnits,
      numLayers = numLayers,
      numResidualLayers = numResLayers,
      dropout = dropout,
      residualFn = residualFn,
      seed = context.env.randomSeed,
      name = "StackedUniCell")
    val initialState = encodedSequences.rnnTuple.state
    (uniCell, initialState)
  }
}

class UnidirectionalRNNDecoderWithAttention[T: TF : IsNotQuantized, State: OutputStructure, AttentionState: OutputStructure, StateShape, AttentionStateShape](
    val cell: Cell[T, State, StateShape],
    val numUnits: Int,
    val numLayers: Int,
    val attention: RNNAttention[T, AttentionState, AttentionStateShape],
    val residual: Boolean = false,
    val dropout: Option[Float] = None,
    val residualFn: Option[(Output[T], Output[T]) => Output[T]] = None,
    val outputAttention: Boolean = true
)(implicit
    evOutputToShapeState: OutputToShape.Aux[State, StateShape],
    evOutputToShapeAttentionState: OutputToShape.Aux[AttentionState, AttentionStateShape]
) extends RNNDecoder[T, State, AttentionWrapperState[T, Seq[State], AttentionState], (Seq[StateShape], Shape, Shape, Seq[Shape], Seq[Shape], Seq[Attention.StateShape[AttentionStateShape]])] {
  override def embeddings(
      ids: Output[Int]
  )(implicit context: Context): Output[T] = {
    val embeddingsTable = context.parameterManager.wordEmbeddings(context.tgtLanguageID)
    embeddingsTable(ids).castTo[T]
  }

  override protected def cellAndInitialState(
      encodedSequences: EncodedSequences[T, State],
      tgtSequences: Option[Sequences[Int]]
  )(implicit context: Context): (AttentionWrapperCell[T, Seq[State], AttentionState, Seq[StateShape], AttentionStateShape], AttentionWrapperState[T, Seq[State], AttentionState]) = {
    val numResLayers = if (residual && numLayers > 1) numLayers - 1 else 0
    val uniCell = stackedCell[T, State, StateShape](
      cell = cell,
      numInputs = 2 * numUnits,
      numUnits = numUnits,
      numLayers = numLayers,
      numResidualLayers = numResLayers,
      dropout = dropout,
      residualFn = residualFn,
      seed = context.env.randomSeed,
      name = "StackedUniCell")
    val memory = Sequences(
      sequences = {
        if (context.modelConfig.timeMajor)
          encodedSequences.rnnTuple.output.transpose(Tensor(1, 0, 2))
        else
          encodedSequences.rnnTuple.output
      },
      lengths = encodedSequences.lengths)
    val attentionCell = attention.createCell(uniCell, memory, numUnits, numUnits, useAttentionLayer = true, outputAttention)
    val initialState = attentionCell.initialState(encodedSequences.rnnTuple.state)
    (attentionCell, initialState)
  }
}
