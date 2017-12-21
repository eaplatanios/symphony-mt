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

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.rnn.cell.{CellInstance, RNNCell}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.attention.AttentionWrapperState
import org.platanios.tensorflow.api.ops.variables.OnesInitializer

/**
  * @author Emmanouil Antonios Platanios
  */
trait Attention {
  type AttentionCellInstance[S, SS] = CellInstance[
      Output, Shape, AttentionWrapperState[Seq[S], Seq[SS]], (Seq[SS], Shape, Shape, Seq[Shape], Seq[Shape])]
  type AttentionInitialState[S, SS] = AttentionWrapperState[Seq[S], Seq[SS]]

  def create[S, SS](
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
  ): (AttentionCellInstance[S, SS], AttentionInitialState[S, SS])
}

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
  ): (AttentionCellInstance[S, SS], AttentionInitialState[S, SS]) = {
    val memoryWeights = tf.variable("MemoryWeights", memory.dataType, Shape(memory.shape(-1), numUnits), null)
    val scale = if (scaled) tf.variable("LuongFactor", memory.dataType, Shape.scalar(), OnesInitializer) else null
    val attention = tf.LuongAttention(
      memory, memoryWeights.value, memorySequenceLengths, scale.value, probabilityFn, scoreMask, "Attention")
    val attentionWeights = tf.variable(
      "AttentionWeights", attention.dataType, Shape(numUnits + memory.shape(-1), numUnits), null)
    val cellInstance = cell.createCell(mode, Shape(inputSequencesLastAxisSize + numUnits))
    val attentionCell = tf.AttentionWrapperCell(
      cellInstance.cell, Seq(attention), Seq(attentionWeights.value), outputAttention = outputAttention)
    val attentionCellInstance = CellInstance(
      cell = attentionCell, trainableVariables = cellInstance.trainableVariables + attentionWeights,
      nonTrainableVariables = cellInstance.nonTrainableVariables)
    (attentionCellInstance, attentionCell.initialState(initialState, memory.dataType))
  }
}
