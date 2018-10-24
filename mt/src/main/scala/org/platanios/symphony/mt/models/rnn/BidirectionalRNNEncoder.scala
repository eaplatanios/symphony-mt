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
import org.platanios.symphony.mt.models.{DeviceManager, RNNModel, Stage}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape, Zero}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple

/** Bi-directional RNN encoder.
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
  override def create(
      config: RNNModel.Config[T, _],
      srcSequences: Output[Int],
      srcSequenceLengths: Output[Int]
  )(implicit
      stage: Stage,
      mode: Mode,
      env: Environment,
      parameterManager: ParameterManager,
      deviceManager: DeviceManager,
      context: Output[Int]
  ): Tuple[Output[T], Seq[State]] = {
    val (embeddedSequences, embeddedSequenceLengths) = embedSequences(config, srcSequences, srcSequenceLengths)
    val numResLayers = if (residual && numLayers > 1) numLayers - 1 else 0

    // Build the forward RNN cell.
    val biCellFw = RNNModel.stackedCell[T, State, StateShape](
      cell = cell,
      numInputs = embeddedSequences.shape(-1),
      numUnits = numUnits,
      numLayers = numLayers / 2,
      numResidualLayers = numResLayers / 2,
      dropout = dropout,
      residualFn = residualFn,
      seed = config.env.randomSeed,
      name = "MultiBiCellFw")

    // Build the backward RNN cell.
    val biCellBw = RNNModel.stackedCell[T, State, StateShape](
      cell = cell,
      numInputs = embeddedSequences.shape(-1),
      numUnits = numUnits,
      numLayers = numLayers / 2,
      numResidualLayers = numResLayers / 2,
      dropout = dropout,
      residualFn = residualFn,
      seed = config.env.randomSeed,
      name = "MultiBiCellBw")

    val unmergedBiTuple = tf.bidirectionalDynamicRNN(
      cellFw = biCellFw,
      cellBw = biCellBw,
      input = embeddedSequences,
      initialStateFw = None,
      initialStateBw = None,
      timeMajor = config.timeMajor,
      parallelIterations = config.env.parallelIterations,
      swapMemory = config.env.swapMemory,
      sequenceLengths = embeddedSequenceLengths,
      name = "BidirectionalLayers")

    Tuple(
      // The bidirectional RNN output is the concatenation of the forward and the backward RNN outputs.
      output = tf.concatenate(Seq(unmergedBiTuple._1.output, unmergedBiTuple._2.output), -1),
      state = unmergedBiTuple._1.state.map(List(_))
          .zipAll(unmergedBiTuple._2.state.map(List(_)), Nil, Nil)
          .flatMap(Function.tupled(_ ++ _)))
  }
}

object BidirectionalRNNEncoder {
  def apply[T: TF : IsNotQuantized, State: OutputStructure, StateShape](
      cell: Cell[T, State, StateShape],
      numUnits: Int,
      numLayers: Int,
      residual: Boolean = false,
      dropout: Option[Float] = None,
      residualFn: Option[(Output[T], Output[T]) => Output[T]] = None
  )(implicit
      evOutputToShapeState: OutputToShape.Aux[State, StateShape],
      evZeroState: Zero.Aux[State, StateShape]
  ): BidirectionalRNNEncoder[T, State, StateShape] = {
    new BidirectionalRNNEncoder[T, State, StateShape](
      cell, numUnits, numLayers, residual, dropout, residualFn)
  }
}
