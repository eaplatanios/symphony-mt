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

import org.platanios.tensorflow.api._

/** The attention prepend mode is a mechanism that allows us to optionally treat a sequence-to-sequence model as a
  * language model. It allows us to prepend the inputs to the targets, while performing decoding.
  *
  * @author Emmanouil Antonios Platanios
  */
trait AttentionPrependMode {
  /** Creates a bias tensor to be added to the attention logits.
    *
    * @param  padding Tensor with shape `[batchSize, length]`, with ones in positions corresponding to
    *                 padding and zeros otherwise. In each row of this tensor, a single padding position separates the
    *                 input part from the target part.
    * @return Tensor with shape `[batchSize, 1, length, length]`.
    */
  def apply(padding: Output[Float]): Output[Float]
}

/** Creates a bias tensor to be added to the attention logits, that allows for full connectivity in the "inputs" part of
  * the sequence and for masked connectivity in the targets part. */
case object AttentionPrependInputsFullAttention extends AttentionPrependMode {
  /** Creates a bias tensor to be added to the attention logits.
    *
    * @param  padding Tensor with shape `[batchSize, length]`, with ones in positions corresponding to
    *                 padding and zeros otherwise. In each row of this tensor, a single padding position separates the
    *                 input part from the target part.
    * @return Tensor with shape `[batchSize, 1, length, length]`.
    */
  override def apply(padding: Output[Float]): Output[Float] = {
    // Everything past the first padding position is part of the target. This tensor has zeros for the source portion
    // and the separator, and ones for the target portion.
    val inTarget = tf.cumsum(padding, axis = 1, exclusive = true)
    // Position within the target, or 0 if part of the source.
    val targetPositions = tf.cumsum(inTarget, axis = 1)
    // A position with a lesser `targetPosition` cannot see a position with a greater `targetPosition`.
    val illegalConnections = tf.greater(
      tf.expandDims(targetPositions, axis = 1),
      tf.expandDims(targetPositions, axis = 2))
    tf.expandDims(illegalConnections.toFloat * -1e9f, axis = 1)
  }
}

/** Creates a bias tensor to be added to the attention logits, that allows allows a query to attend to all positions up
  * to and including its own. */
case object AttentionPrependInputsMaskedAttention extends AttentionPrependMode {
  /** Creates a bias tensor to be added to the attention logits.
    *
    * @param  padding Tensor with shape `[batchSize, length]`, with ones in positions corresponding to
    *                 padding and zeros otherwise. In each row of this tensor, a single padding position separates the
    *                 input part from the target part.
    * @return Tensor with shape `[1, 1, length, length]`.
    */
  override def apply(padding: Output[Float]): Output[Float] = {
    Attention.attentionBiasLowerTriangular(tf.shape(padding).slice(1).toInt)
  }
}
