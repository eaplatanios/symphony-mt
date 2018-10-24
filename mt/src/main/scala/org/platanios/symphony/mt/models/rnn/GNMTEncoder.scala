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

/**
  * @author Emmanouil Antonios Platanios
  */
class GNMTEncoder[T: TF : IsNotQuantized, State: OutputStructure, StateShape](
    val cell: Cell[T, State, StateShape],
    val numUnits: Int,
    val numBiLayers: Int,
    val numUniLayers: Int,
    val numUniResLayers: Int,
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

    // Bidirectional RNN layers
    val biTuple = {
      if (numBiLayers > 0) {
        val biCellFw = RNNModel.stackedCell[T, State, StateShape](
          cell = cell,
          numInputs = embeddedSequences.shape(-1),
          numUnits = numUnits,
          numLayers = numBiLayers,
          numResidualLayers = 0,
          dropout = dropout,
          residualFn = residualFn,
          seed = config.env.randomSeed,
          name = "MultiBiCellFw")
        val biCellBw = RNNModel.stackedCell[T, State, StateShape](
          cell = cell,
          numInputs = embeddedSequences.shape(-1),
          numUnits = numUnits,
          numLayers = numBiLayers,
          numResidualLayers = 0,
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
        Tuple(tf.concatenate(Seq(unmergedBiTuple._1.output, unmergedBiTuple._2.output), -1), unmergedBiTuple._2.state)
      } else {
        Tuple(embeddedSequences, Seq.empty[State])
      }
    }

    // Unidirectional RNN layers
    val uniCell = RNNModel.stackedCell[T, State, StateShape](
      cell = cell,
      numInputs = biTuple.output.shape(-1),
      numUnits = numUnits,
      numLayers = numUniLayers,
      numResidualLayers = numUniResLayers,
      dropout = dropout,
      residualFn = residualFn,
      seed = config.env.randomSeed,
      name = "MultiUniCell")
    val uniTuple = tf.dynamicRNN(
      cell = uniCell,
      input = biTuple.output,
      initialState = None,
      timeMajor = config.timeMajor,
      parallelIterations = config.env.parallelIterations,
      swapMemory = config.env.swapMemory,
      sequenceLengths = embeddedSequenceLengths,
      name = "UnidirectionalLayers")

    // Pass all of the encoder's state except for the first bi-directional layer's state, to the decoder.
    Tuple(uniTuple.output, biTuple.state ++ uniTuple.state)
  }
}

object GNMTEncoder {
  def apply[T: TF : IsNotQuantized, State: OutputStructure, StateShape](
      cell: Cell[T, State, StateShape],
      numUnits: Int,
      numBiLayers: Int,
      numUniLayers: Int,
      numUniResLayers: Int,
      dropout: Option[Float] = None,
      residualFn: Option[(Output[T], Output[T]) => Output[T]] = None
  )(implicit
      evOutputToShapeState: OutputToShape.Aux[State, StateShape],
      evZeroState: Zero.Aux[State, StateShape]
  ): GNMTEncoder[T, State, StateShape] = {
    new GNMTEncoder[T, State, StateShape](
      cell, numUnits, numBiLayers, numUniLayers, numUniResLayers, dropout, residualFn)
  }
}
