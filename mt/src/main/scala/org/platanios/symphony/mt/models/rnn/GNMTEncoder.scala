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

/**
  * @author Emmanouil Antonios Platanios
  */
class GNMTEncoder[T: TF : IsNotQuantized, State: OutputStructure, StateShape](
    val cell: Cell[T, State, StateShape],
    val numUnits: Int,
    val numBiLayers: Int,
    val numUniLayers: Int,
    val numUniResLayers: Int,
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

    // Bidirectional RNN layers
    val biTuple = {
      if (numBiLayers > 0) {
        val biCellFw = stackedCell[T, State, StateShape](
          cell = cell,
          numInputs = embeddedSequences.sequences.shape(-1),
          numUnits = numUnits,
          numLayers = numBiLayers,
          numResidualLayers = 0,
          dropout = dropout,
          residualFn = residualFn,
          seed = context.env.randomSeed,
          name = "StackedBiCellFw")
        val biCellBw = stackedCell[T, State, StateShape](
          cell = cell,
          numInputs = embeddedSequences.sequences.shape(-1),
          numUnits = numUnits,
          numLayers = numBiLayers,
          numResidualLayers = 0,
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
        RNNTuple(tf.concatenate(Seq(unmergedBiTuple._1.output, unmergedBiTuple._2.output), -1), unmergedBiTuple._2.state)
      } else {
        RNNTuple(embeddedSequences.sequences.castTo[T], Seq.empty[State])
      }
    }

    // Unidirectional RNN layers
    val uniCell = stackedCell[T, State, StateShape](
      cell = cell,
      numInputs = biTuple.output.shape(-1),
      numUnits = numUnits,
      numLayers = numUniLayers,
      numResidualLayers = numUniResLayers,
      dropout = dropout,
      residualFn = residualFn,
      seed = context.env.randomSeed,
      name = "StackedUniCell")
    val uniTuple = tf.dynamicRNN(
      cell = uniCell,
      input = biTuple.output,
      initialState = None,
      timeMajor = context.modelConfig.timeMajor,
      parallelIterations = context.env.parallelIterations,
      swapMemory = context.env.swapMemory,
      sequenceLengths = embeddedSequences.lengths,
      name = "UnidirectionalLayers")

    // Pass all of the encoder's state except for the first bi-directional layer's state, to the decoder.
    val rnnTuple = RNNTuple(uniTuple.output, biTuple.state ++ uniTuple.state)

    EncodedSequences(rnnTuple, embeddedSequences.lengths)
  }
}
