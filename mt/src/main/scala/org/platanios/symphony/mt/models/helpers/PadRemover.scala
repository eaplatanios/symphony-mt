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

package org.platanios.symphony.mt.models.helpers

import org.platanios.tensorflow.api._

import scala.language.postfixOps

/** Helper to remove padding from a tensor before applying a computation to it.
  *
  * The padding is computed for one reference tensor containing the padding mask and can then be applied to any other
  * tensor with a compatible shape.
  *
  * @param  padMask Reference padding tensor with shape `[batchSize, length]` or `[originAxisSize]` (where
  *                 `originAxisSize = batchSize * length`) containing non-zero positive values to indicate the padding
  *                 locations.
  * @param  name    Name for the created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
case class PadRemover(
    padMask: Output[Float],
    name: String = "PadRemover"
) {
  /** `nonPadIndices` contains coordinates of zero rows (as `padMask` may be `FLOAT32`, checking zero equality is done
    * with `|x| < epsilon`, with `epsilon = 1e-9` as standard. Here padMask contains only positive values and so the
    * absolute value is not needed. */
  val (nonPadIndices, originAxisSize) = {
    tf.nameScope(s"$name/Initialization") {
      val flattenedPadMask = padMask.reshape(Shape(-1))
      (tf.where(flattenedPadMask < 1e-9f).toInt, tf.shape(flattenedPadMask).slice(0 :: 1))
    }
  }

  /** Removes padding from the provided `value`.
    *
    * @param  value Tensor with shape `[originAxisSize, ...]`.
    * @return Tensor with shape `[originAxisSizeCompressed, ...]`, where `originAxisSizeCompressed <= originAxisSize`.
    */
  def remove(
      value: Output[Float]
  ): Output[Float] = {
    tf.nameScope(s"$name/Remove") {
      val valueShape = value.shape
      val result = tf.gatherND(value, nonPadIndices)
      result.setShape(Shape(-1) ++ valueShape(1 ::))
      result
    }
  }

  /** Adds padding back to the provided `value`.
    *
    * @param  value Tensor with shape `[originAxisSizeCompressed, ...]`, where
    *               `originAxisSizeCompressed <= originAxisSize`.
    * @return Tensor with shape `[originAxisSize, ...]`.
    */
  def restore(
      value: Output[Float]
  ): Output[Float] = {
    tf.nameScope(s"$name/Add") {
      tf.scatterND(nonPadIndices, value, tf.concatenate(Seq(originAxisSize, tf.shape(value).slice(1 ::))))
    }
  }
}
