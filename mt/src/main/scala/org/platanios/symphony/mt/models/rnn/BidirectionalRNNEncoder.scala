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

import org.platanios.symphony.mt.models.{Context, Sequences}
import org.platanios.symphony.mt.models.Utilities._
import org.platanios.symphony.mt.models.rnn.Utilities._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.tf.RNNTuple
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape, Zero}

/** Bi-directional (i.e., left-to-right) RNN encoder.
  *
  * This encoder takes as input a source sequence in some language and returns a tuple containing:
  *   - '''Output:''' Concatenated outputs (for each time step) of the forward RNN and the backward RNN.
  *   - '''State:''' Sequence of last computed RNN states in layer order containing both the forward and the backward
  *     states for each layer (e.g., `Seq(forwardState0, backwardState0, forwardState1, backwardState1, ...)`).
  *
  * @author Emmanouil Antonios Platanios
  */
class BidirectionalRNNEncoder[T: TF : IsNotQuantized, State: OutputStructure, StateShape](
    val cell: Cell[T, State, StateShape],
    val numUnits: Int,
    val numLayers: Int,
    val residual: Boolean = false,
    val dropout: Option[Float] = None,
    val residualFn: Option[(Output[T], Output[T]) => Output[T]] = None
)(implicit
    evOutputToShapeState: OutputToShape.Aux[State, StateShape],
    evZeroState: Zero.Aux[State, StateShape]
) extends RNNEncoder[T, State]() {
  override def apply(
      sequences: Sequences[Int]
  )(implicit context: Context): EncodedSequences[T, State] = {
    val embeddedSequences = maybeTransposeInputSequences(embedSequences(sequences))
    val numResLayers = if (residual && numLayers > 1) numLayers - 1 else 0

    // Build the forward RNN cell.
    val biCellFw = stackedCell[T, State, StateShape](
      cell = cell,
      numInputs = embeddedSequences.sequences.shape(-1),
      numUnits = numUnits,
      numLayers = numLayers / 2,
      numResidualLayers = numResLayers / 2,
      dropout = dropout,
      residualFn = residualFn,
      seed = context.env.randomSeed,
      name = "StackedBiCellFw")

    // Build the backward RNN cell.
    val biCellBw = stackedCell[T, State, StateShape](
      cell = cell,
      numInputs = embeddedSequences.sequences.shape(-1),
      numUnits = numUnits,
      numLayers = numLayers / 2,
      numResidualLayers = numResLayers / 2,
      dropout = dropout,
      residualFn = residualFn,
      seed = context.env.randomSeed,
      name = "StackedBiCellBw")

    val unmergedBiTuple = tf.bidirectionalDynamicRNN(
      cellFw = biCellFw,
      cellBw = biCellBw,
      input = embeddedSequences.sequences.castTo[T],
      initialStateFw = None,
      initialStateBw = None,
      timeMajor = context.modelConfig.timeMajor,
      parallelIterations = context.env.parallelIterations,
      swapMemory = context.env.swapMemory,
      sequenceLengths = embeddedSequences.lengths,
      name = "BidirectionalLayers")

    val rnnTuple = RNNTuple(
      // The bidirectional RNN output is the concatenation of the forward and the backward RNN outputs.
      output = tf.concatenate(Seq(unmergedBiTuple._1.output, unmergedBiTuple._2.output), axis = -1),
      state = unmergedBiTuple._1.state.map(List(_))
          .zipAll(unmergedBiTuple._2.state.map(List(_)), Nil, Nil)
          .flatMap(Function.tupled(_ ++ _)))

    EncodedSequences(rnnTuple, embeddedSequences.lengths)
  }
}
