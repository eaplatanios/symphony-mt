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

import org.platanios.symphony.mt.models.Context
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape}

/**
  * @author Emmanouil Antonios Platanios
  */
object Utilities {
  def cell[T: TF : IsNotQuantized, State: OutputStructure, StateShape](
      cell: Cell[T, State, StateShape],
      numInputs: Int,
      numUnits: Int,
      dropout: Float = 0.0f,
      residualFn: Option[(Output[T], Output[T]) => Output[T]] = None,
      device: String = "",
      seed: Option[Int] = None,
      name: String
  )(implicit
      context: Context,
      evOutputToShape: OutputToShape.Aux[State, StateShape]
  ): tf.RNNCell[Output[T], State, Shape, StateShape] = {
    tf.variableScope(name) {
      tf.createWith(device = device) {
        // Create the main RNN cell.
        var createdCell = cell.create(name, numInputs, numUnits)

        // Optionally, apply dropout.
        if (dropout > 0.0f) {
          createdCell = tf.DropoutWrapper(createdCell, 1.0f - dropout, seed = seed, name = "Dropout")
        }

        // Add residual connections.
        createdCell = residualFn.map(tf.ResidualWrapper(createdCell, _)).getOrElse(createdCell)

        createdCell
      }
    }
  }

  def cells[T: TF : IsNotQuantized, State: OutputStructure, StateShape](
      cell: Cell[T, State, StateShape],
      numInputs: Int,
      numUnits: Int,
      numLayers: Int,
      numResidualLayers: Int,
      dropout: Float = 0.0f,
      residualFn: Option[(Output[T], Output[T]) => Output[T]] = None,
      seed: Option[Int] = None,
      name: String
  )(implicit
      context: Context,
      evOutputToShape: OutputToShape.Aux[State, StateShape]
  ): Seq[tf.RNNCell[Output[T], State, Shape, StateShape]] = {
    tf.variableScope(name) {
      (0 until numLayers).foldLeft(Seq.empty[tf.RNNCell[Output[T], State, Shape, StateShape]])((cells, i) => {
        val cellNumInputs = if (i == 0) numInputs else cells(i - 1).outputShape(-1)
        cells :+ this.cell[T, State, StateShape](
          cell, cellNumInputs, numUnits, dropout,
          if (i >= numLayers - numResidualLayers) residualFn else None,
          context.nextDevice(), seed, s"Cell$i")
      })
    }
  }

  def stackedCell[T: TF : IsNotQuantized, State: OutputStructure, StateShape](
      cell: Cell[T, State, StateShape],
      numInputs: Int,
      numUnits: Int,
      numLayers: Int,
      numResidualLayers: Int,
      dropout: Float = 0.0f,
      residualFn: Option[(Output[T], Output[T]) => Output[T]] = None,
      seed: Option[Int] = None,
      name: String
  )(implicit
      context: Context,
      evOutputToShape: OutputToShape.Aux[State, StateShape]
  ): tf.RNNCell[Output[T], Seq[State], Shape, Seq[StateShape]] = {
    tf.StackedCell(cells[T, State, StateShape](
      cell, numInputs, numUnits, numLayers, numResidualLayers, dropout,
      residualFn, seed, name), name)
  }
}
