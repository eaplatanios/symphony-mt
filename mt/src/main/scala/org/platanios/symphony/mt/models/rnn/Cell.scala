/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.learn.layers.rnn.cell.{GRUCell, LSTMCell, RNNCell}
import org.platanios.tensorflow.api.ops.rnn.cell.LSTMState
import org.platanios.tensorflow.api.ops.{Math, Output}
import org.platanios.tensorflow.api.types.DataType

/**
  * @author Emmanouil Antonios Platanios
  */
trait Cell[S, SS] {
  def create(name: String, numUnits: Int, dataType: DataType): RNNCell[Output, Shape, S, SS]
}

case class GRU(activation: Output => Output = Math.tanh(_)) extends Cell[Output, Shape] {
  override def create(name: String, numUnits: Int, dataType: DataType): RNNCell[Output, Shape, Output, Shape] = {
    GRUCell(name, numUnits, dataType, activation)
  }
}

case class LSTM(
    forgetBias: Float = 1.0f, usePeepholes: Boolean = false, cellClip: Float = -1,
    projectionSize: Int = -1, projectionClip: Float = -1, activation: Output => Output = Math.tanh(_)
) extends Cell[LSTMState, (Shape, Shape)] {
  override def create(name: String, numUnits: Int, dataType: DataType): LSTMCell = {
    LSTMCell(name, numUnits, dataType, forgetBias, usePeepholes, cellClip, projectionSize, projectionClip, activation)
  }
}
