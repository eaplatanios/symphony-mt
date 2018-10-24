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

package org.platanios.symphony.mt.models.attention

import org.platanios.symphony.mt.models.helpers.Common
import org.platanios.symphony.mt.models.parameters.ParameterManager
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsHalfOrFloatOrDouble, TF}
import org.platanios.tensorflow.api.learn.Mode

/** Dot-product attention.
  *
  * @param  dropoutRate          Dropout rate for the attention weights.
  * @param  dropoutBroadcastAxes Specifies along which axes of the attention weights the dropout is broadcast.
  * @param  name                 Name for this attention model that also specifies a variable scope.
  *
  * @author Emmanouil Antonios Platanios
  */
class DotProductAttention protected (
    val dropoutRate: Float = 0.0f,
    val dropoutBroadcastAxes: Set[Int] = Set.empty,
    val name: String = "DotProductAttention"
) extends Attention {
  /** Computes the attention for the provided queries, keys, and values.
    *
    * @param  q                 Queries tensor with shape `[batchSize, ..., length, depth]`.
    * @param  k                 Keys tensor with shape `[batchSize, ..., length, depth]`.
    * @param  v                 Values tensor with shape `[batchSize, ..., length, depth]`.
    * @param  bias              Optional attention bias.
    * @param  mode              Current learning mode (e.g., training or evaluation).
    * @param  parameterManager Parameter manager to use, if parameters are required.
    * @return Attention tensor with shape `[batchSize, ..., length, depth]`.
    */
  override def apply[T: TF : IsHalfOrFloatOrDouble](
      q: Output[T],
      k: Output[T],
      v: Output[T],
      bias: Option[Output[T]]
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager
  ): Output[T] = {
    tf.nameScope(name) {
      // `logits` shape: [batchSize, numHeads, queryLength, memoryLength]
      var logits = tf.matmul(q, k, transposeB = true)
      bias.foreach(logits += _)
      var weights = tf.softmax(logits, name = "AttentionWeights")
      // Apply dropout to the attention links for each of the heads.
      if (mode.isTraining)
        weights = Common.dropoutWithBroadcastAxes(weights, 1.0f - dropoutRate, broadcastAxes = dropoutBroadcastAxes)
      tf.matmul(weights, v)
    }
  }
}

object DotProductAttention {
  def apply(
      dropoutRate: Float = 0.0f,
      dropoutBroadcastAxes: Set[Int] = Set.empty,
      name: String = "DotProductAttention"
  ): DotProductAttention = {
    new DotProductAttention(dropoutRate, dropoutBroadcastAxes, name)
  }
}
