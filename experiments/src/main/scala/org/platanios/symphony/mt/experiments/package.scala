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

package org.platanios.symphony.mt

import org.platanios.symphony.mt.models.rnn.Cell
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape, Zero}

/**
  * @author Emmanouil Antonios Platanios
  */
package object experiments {
  implicit def cellToEvOutputStructureState[T](
      cell: Cell[T, _, _]
  ): OutputStructure[cell.StateType] = {
    cell.evOutputStructureState.asInstanceOf[OutputStructure[cell.StateType]]
  }

  implicit def cellToEvOutputToShapeState[T](
      cell: Cell[T, _, _]
  ): OutputToShape.Aux[cell.StateType, cell.StateShapeType] = {
    cell.evOutputToShapeState.asInstanceOf[OutputToShape.Aux[cell.StateType, cell.StateShapeType]]
  }

  implicit def cellToEvZeroState[T](
      cell: Cell[T, _, _]
  ): Zero.Aux[cell.StateType, cell.StateShapeType] = {
    cell.evZeroState.asInstanceOf[Zero.Aux[cell.StateType, cell.StateShapeType]]
  }
}
