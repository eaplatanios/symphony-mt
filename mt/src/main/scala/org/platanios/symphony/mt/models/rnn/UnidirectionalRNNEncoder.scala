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
import org.platanios.tensorflow.api.implicits.helpers.{NestedStructure, Zero}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple

/** Uni-directional (i.e., left-to-right) RNN encoder.
  *
  * This encoder takes as input a source sequence in some language and returns a tuple containing:
  *   - '''Output:''' Outputs (for each time step) of the RNN.
  *   - '''State:''' Sequence of last computed RNN states in layer order containing the states for each layer
  *     (e.g., `Seq(state0, state1, ...)`).
  *
  * @author Emmanouil Antonios Platanios
  */
class UnidirectionalRNNEncoder[T: TF : IsNotQuantized, State](
    val cell: Cell[T, State],
    val numUnits: Int,
    val numLayers: Int,
    val residual: Boolean = false,
    val dropout: Option[Float] = None,
    val residualFn: Option[(Output[T], Output[T]) => Output[T]] = None
)(implicit
    override val evZeroState: Zero[State]
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
    implicit val evZeroState: Zero.Aux[State, _, _, _] = this.evZeroState.asAux()
    implicit val evStructureState: NestedStructure.Aux[State, _, _, _] = evZeroState.structure

    val (embeddedSequences, embeddedSequenceLengths) = embedSequences(config, srcSequences, srcSequenceLengths)
    val numResLayers = if (residual && numLayers > 1) numLayers - 1 else 0

    val uniCell = RNNModel.stackedCell[T, State](
      cell = cell,
      numInputs = embeddedSequences.shape(-1),
      numUnits = numUnits,
      numLayers = numLayers,
      numResidualLayers = numResLayers,
      dropout = dropout,
      residualFn = residualFn,
      seed = config.env.randomSeed,
      name = "MultiUniCell")

    tf.dynamicRNN(
      cell = uniCell,
      input = embeddedSequences,
      initialState = None,
      timeMajor = config.timeMajor,
      parallelIterations = config.env.parallelIterations,
      swapMemory = config.env.swapMemory,
      sequenceLengths = embeddedSequenceLengths,
      name = "UnidirectionalLayers")
  }
}

object UnidirectionalRNNEncoder {
  def apply[T: TF : IsNotQuantized, State: Zero](
      cell: Cell[T, State],
      numUnits: Int,
      numLayers: Int,
      residual: Boolean = false,
      dropout: Option[Float] = None,
      residualFn: Option[(Output[T], Output[T]) => Output[T]] = None
  ): UnidirectionalRNNEncoder[T, State] = {
    new UnidirectionalRNNEncoder[T, State](
      cell, numUnits, numLayers, residual, dropout, residualFn)
  }
}
