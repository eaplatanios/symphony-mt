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

package org.platanios.symphony.mt.models.attention

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.rnn.cell.RNNCell
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.attention.{AttentionWrapperCell, AttentionWrapperState}
import org.platanios.tensorflow.api.ops.variables.OnesInitializer
import org.platanios.tensorflow.api.{ops, tf}

/**
  * @author Emmanouil Antonios Platanios
  */
case class LuongAttention(
    scaled: Boolean = false,
    probabilityFn: (Output) => Output = tf.softmax(_, name = "Probability"),
    scoreMask: Float = Float.NegativeInfinity
) extends Attention {
  override def create[S, SS](
      cell: RNNCell[Output, Shape, Seq[S], Seq[SS]],
      memory: Output,
      memorySequenceLengths: Output,
      numUnits: Int,
      inputSequencesLastAxisSize: Int,
      initialState: Seq[S],
      outputAttention: Boolean,
      mode: Mode
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): (AttentionWrapperCell[Seq[S], Seq[SS]], AttentionWrapperState[Seq[S], Seq[SS]]) = {
    val memoryWeights = tf.variable("MemoryWeights", memory.dataType, Shape(memory.shape(-1), numUnits), null)
    val scale = if (scaled) tf.variable("LuongFactor", memory.dataType, Shape.scalar(), OnesInitializer) else null
    val attention = tf.LuongAttention(
      memory, memoryWeights.value, memorySequenceLengths, scale.value, probabilityFn, scoreMask, "Attention")
    val attentionWeights = tf.variable(
      "AttentionWeights", attention.dataType, Shape(numUnits + memory.shape(-1), numUnits), null)
    val createdCell = cell.createCell(mode, Shape(inputSequencesLastAxisSize + numUnits))
    val attentionCell = tf.AttentionWrapperCell(
      createdCell, Seq(attention), Seq(attentionWeights.value), outputAttention = outputAttention)
    (attentionCell, attentionCell.initialState(initialState, memory.dataType))
  }
}
