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

import org.platanios.symphony.mt.models.Stage
import org.platanios.symphony.mt.models.parameters.ParameterManager
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsDecimal, TF}
import org.platanios.tensorflow.api.implicits.helpers.NestedStructure
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.rnn.attention.{AttentionWrapperCell, AttentionWrapperState}
import org.platanios.tensorflow.api.ops.variables.{ConstantInitializer, ZerosInitializer}

/**
  * @author Emmanouil Antonios Platanios
  */
case class BahdanauRNNAttention[T: TF : IsDecimal](
    normalized: Boolean = false,
    probabilityFn: Output[T] => Output[T], // TODO: Softmax should be the default.
    scoreMask: Float = Float.NegativeInfinity.toFloat
) extends RNNAttention[T, Output[T]] {
  override def create[CellState: NestedStructure](
      cell: tf.RNNCell[Output[T], CellState],
      memory: Output[T],
      memorySequenceLengths: Output[Int],
      numUnits: Int,
      inputSequencesLastAxisSize: Int,
      initialState: CellState,
      useAttentionLayer: Boolean,
      outputAttention: Boolean
  )(implicit
      stage: Stage,
      mode: Mode,
      parameterManager: ParameterManager,
      context: Output[Int],
  ): (AttentionWrapperCell[T, CellState, Output[T]],
      AttentionWrapperState[T, CellState, Seq[Output[T]]]) = {
    tf.variableScope("BahdanauAttention") {
      val memoryWeights = parameterManager.get[T]("MemoryWeights", Shape(numUnits, numUnits))
      val queryWeights = parameterManager.get[T]("QueryWeights", Shape(numUnits, numUnits))
      val scoreWeights = parameterManager.get[T]("ScoreWeights", Shape(numUnits))
      val (normFactor, normBias) = {
        if (normalized) {
          (parameterManager.get[T]("Factor", Shape(), ConstantInitializer(math.sqrt(1.0f / numUnits).toFloat)),
              parameterManager.get[T]("Bias", Shape(numUnits), ZerosInitializer))
        } else {
          (null, null)
        }
      }
      val attention = tf.BahdanauAttention(
        memory, memoryWeights, queryWeights, scoreWeights, probabilityFn,
        memorySequenceLengths, normFactor, normBias, scoreMask, "Attention")
      val attentionWeights = {
        if (useAttentionLayer)
          Seq(tf.variable[T]("AttentionWeights", Shape(numUnits + memory.shape(-1), numUnits), null).value)
        else
          null
      }
      val attentionCell = tf.AttentionWrapperCell(
        cell, Seq(attention), attentionWeights, outputAttention = outputAttention)
      (attentionCell, attentionCell.initialState(initialState))
    }
  }
}
