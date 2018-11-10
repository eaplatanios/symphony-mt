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
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape, Zero}

/** Uni-directional (i.e., left-to-right) RNN encoder.
  *
  * This encoder takes as input a source sequence in some language and returns a tuple containing:
  *   - '''Output:''' Outputs (for each time step) of the RNN.
  *   - '''State:''' Sequence of last computed RNN states in layer order containing the states for each layer
  *     (e.g., `Seq(state0, state1, ...)`).
  *
  * @author Emmanouil Antonios Platanios
  */
class UnidirectionalRNNEncoder[T: TF : IsNotQuantized, State: OutputStructure, StateShape](
    val cell: Cell[T, State, StateShape],
    val numUnits: Int,
    val numLayers: Int,
    val residual: Boolean = false,
    val dropout: Float = 0.0f,
    val residualFn: Option[(Output[T], Output[T]) => Output[T]] = None
)(implicit
    evOutputToShapeState: OutputToShape.Aux[State, StateShape],
    evZeroState: Zero.Aux[State, StateShape]
) extends RNNEncoder[T, State]() {
  override def apply(
      sequences: Sequences[Int]
  )(implicit context: Context): EncodedSequences[T, State] = {
    val embeddedSequences = maybeTransposeInputSequences(embedSrcSequences(sequences))
    val numResLayers = if (residual && numLayers > 1) numLayers - 1 else 0
    val uniCell = stackedCell[T, State, StateShape](
      cell = cell,
      numInputs = embeddedSequences.sequences.shape(-1),
      numUnits = numUnits,
      numLayers = numLayers,
      numResidualLayers = numResLayers,
      dropout = dropout,
      residualFn = residualFn,
      seed = context.env.randomSeed,
      name = "StackedUniCell")

    val rnnTuple = tf.dynamicRNN(
      cell = uniCell,
      input = embeddedSequences.sequences.castTo[T],
      initialState = None,
      timeMajor = context.modelConfig.timeMajor,
      parallelIterations = context.env.parallelIterations,
      swapMemory = context.env.swapMemory,
      sequenceLengths = embeddedSequences.lengths,
      name = "UnidirectionalLayers")

    EncodedSequences(rnnTuple, embeddedSequences.lengths)
  }
}
