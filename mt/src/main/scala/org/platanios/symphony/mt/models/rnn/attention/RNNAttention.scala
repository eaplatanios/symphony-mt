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

package org.platanios.symphony.mt.models.rnn.attention

import org.platanios.symphony.mt.models.{ModelConstructionContext, Sequences}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape}
import org.platanios.tensorflow.api.ops.rnn.attention.AttentionWrapperCell

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class RNNAttention[T: TF : IsNotQuantized, AttentionState, AttentionStateShape] {
  def createCell[CellState: OutputStructure, CellStateShape](
      cell: tf.RNNCell[Output[T], CellState, Shape, CellStateShape],
      memory: Sequences[T],
      numUnits: Int,
      inputSequencesLastAxisSize: Int,
      useAttentionLayer: Boolean,
      outputAttention: Boolean
  )(implicit
      context: ModelConstructionContext,
      evOutputToShapeCellState: OutputToShape.Aux[CellState, CellStateShape]
  ): AttentionWrapperCell[T, CellState, AttentionState, CellStateShape, AttentionStateShape]
}
