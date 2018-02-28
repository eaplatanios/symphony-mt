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

import org.platanios.symphony.mt.models.ParametersManager
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.attention.{AttentionWrapperCell, AttentionWrapperState}
import org.platanios.tensorflow.api.ops.variables.{ConstantInitializer, ZerosInitializer}

/**
  * @author Emmanouil Antonios Platanios
  */
case class BahdanauRNNAttention(
    normalized: Boolean = false,
    probabilityFn: (Output) => Output = tf.softmax(_, name = "Probability"),
    scoreMask: Float = Float.NegativeInfinity
) extends RNNAttention[Output, Shape] {
  override def create[I, S, SS](
      cell: tf.RNNCell[Output, Shape, S, SS],
      memory: Output,
      memorySequenceLengths: Output,
      numUnits: Int,
      inputSequencesLastAxisSize: Int,
      initialState: S,
      useAttentionLayer: Boolean,
      outputAttention: Boolean
  )(mode: Mode, parametersManager: ParametersManager[I])(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: tf.DropoutWrapper.Supported[S]
  ): (AttentionWrapperCell[S, SS, Output, Shape], AttentionWrapperState[S, SS, Seq[Output], Seq[Shape]]) = {
    tf.createWithVariableScope("BahdanauAttention") {
      val dataType = memory.dataType
      val memoryWeights = parametersManager.get("MemoryWeights", dataType, Shape(numUnits, numUnits))
      val queryWeights = parametersManager.get("QueryWeights", dataType, Shape(numUnits, numUnits))
      val scoreWeights = parametersManager.get("ScoreWeights", dataType, Shape(numUnits))
      val (normFactor, normBias) = {
        if (normalized) {
          (parametersManager.get("Factor", dataType, Shape(), ConstantInitializer(math.sqrt(1.0f / numUnits).toFloat)),
              parametersManager.get("Bias", dataType, Shape(numUnits), ZerosInitializer))
        } else {
          (null, null)
        }
      }
      val attention = tf.BahdanauAttention(
        memory, memoryWeights, queryWeights, scoreWeights, memorySequenceLengths, normFactor, normBias,
        probabilityFn, scoreMask, "Attention")
      val attentionWeights = {
        if (useAttentionLayer)
          Seq(tf.variable(
            "AttentionWeights", attention.dataType, Shape(numUnits + memory.shape(-1), numUnits), null).value)
        else
          null
      }
      val attentionCell = tf.AttentionWrapperCell(
        cell, Seq(attention), attentionWeights, outputAttention = outputAttention)
      (attentionCell, attentionCell.initialState(initialState, dataType))
    }
  }
}
