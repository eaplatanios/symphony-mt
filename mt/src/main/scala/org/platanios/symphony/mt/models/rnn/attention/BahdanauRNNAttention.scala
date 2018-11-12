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
import org.platanios.tensorflow.api.ops.rnn.attention.{Attention, AttentionWrapperCell}
import org.platanios.tensorflow.api.ops.variables.{ConstantInitializer, ZerosInitializer}

/**
  * @author Emmanouil Antonios Platanios
  */
class BahdanauRNNAttention[T: TF : IsDecimal](
    val normalized: Boolean = false,
    val probabilityFn: Output[T] => Output[T], // TODO: Softmax should be the default.
    val scoreMask: Float = Float.NegativeInfinity.toFloat
) extends RNNAttention[T, Output[T], Shape] {
  override def createCell[CellState: OutputStructure, CellStateShape](
      cell: tf.RNNCell[Output[T], CellState, Shape, CellStateShape],
      memory: Sequences[T],
      numUnits: Int,
      inputSequencesLastAxisSize: Int,
      useAttentionLayer: Boolean,
      outputAttention: Boolean
  )(implicit
      context: ModelConstructionContext,
      evOutputToShapeCellState: OutputToShape.Aux[CellState, CellStateShape]
  ): AttentionWrapperCell[T, CellState, Output[T], CellStateShape, Shape] = {
    tf.variableScope("BahdanauAttention") {
      val memoryWeights = context.parameterManager.get[T]("MemoryWeights", Shape(numUnits, numUnits))
      val queryWeights = context.parameterManager.get[T]("QueryWeights", Shape(numUnits, numUnits))
      val scoreWeights = context.parameterManager.get[T]("ScoreWeights", Shape(numUnits))
      val (normFactor, normBias) = {
        if (normalized) {
          (context.parameterManager.get[T]("Factor", Shape(), ConstantInitializer(math.sqrt(1.0f / numUnits).toFloat)),
              context.parameterManager.get[T]("Bias", Shape(numUnits), ZerosInitializer))
        } else {
          (null, null)
        }
      }
      val attention = tf.BahdanauAttention(
        tf.shape(memory.sequences).slice(1), memoryWeights, queryWeights, scoreWeights, probabilityFn,
        normFactor, normBias, scoreMask, "Attention")
      val attentionWeights = {
        if (useAttentionLayer)
          Seq(tf.variable[T]("AttentionWeights", Shape(numUnits + memory.sequences.shape(-1), numUnits), null).value)
        else
          null
      }
      tf.AttentionWrapperCell(
        cell, Seq(Attention.Memory(memory.sequences, Some(memory.lengths)) -> attention),
        attentionWeights, outputAttention = outputAttention)
    }
  }
}
