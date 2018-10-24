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

package org.platanios.symphony.mt.models.helpers

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsIntOrLong, TF}

/** Noam scheduling method, similar to that proposed in:
  * [Attention is All You Need (Section 5.3)](https://arxiv.org/pdf/1706.03762.pdf).
  *
  * This method applies a scheduling function to a provided initial learning rate (i.e., `value`). It requires a
  * step value to be provided in it's application function, in order to compute the decayed learning rate. You may
  * simply pass a TensorFlow variable that you increment at each training step.
  *
  * The decayed value is computed as follows:
  * {{{
  *    decayed = value * 5000.0f * (hiddenSize ** -0.5f) * min((step + 1) * (warmUpSteps ** -1.5f), (step + 1) ** -0.5f)
  * }}}
  *
  * @param  warmUpSteps Number of warm-up steps.
  * @param  hiddenSize  Hidden layers size in the attention model.
  *
  * @author Emmanouil Antonios Platanios
  */
class NoamSchedule protected (
    val warmUpSteps: Int,
    val hiddenSize: Int,
    val name: String = "NoamSchedule"
) extends tf.train.Schedule[Float] {
  /** Applies the scheduling method to `value`, the current iteration in the optimization loop is `step` and returns the
    * result.
    *
    * @param  value Value to change based on this schedule.
    * @param  step  Option containing current iteration in the optimization loop, if one has been provided.
    * @return Potentially modified value.
    * @throws IllegalArgumentException If the scheduling method requires a value for `step` but the provided option is
    *                                  empty.
    */
  @throws[IllegalArgumentException]
  override def apply[I: TF : IsIntOrLong](
      value: Output[Float],
      step: Option[Variable[I]]
  ): Output[Float] = {
    if (step.isEmpty)
      throw new IllegalArgumentException("A step needs to be provided for the Noam scheduling method.")
    tf.nameScope(name) {
      val stepValue = step.get.value.toFloat
      val warmUpStepsValue = tf.constant[Int](warmUpSteps).toFloat
      val hiddenSizeValue = tf.constant[Int](hiddenSize).toFloat
      value * 5000.0f * (hiddenSizeValue ** -0.5f) *
          tf.minimum((stepValue + 1) * (warmUpStepsValue ** -1.5f), (stepValue + 1) ** -0.5f)
    }
  }
}

object NoamSchedule {
  def apply(
      warmUpSteps: Int,
      hiddenSize: Int,
      name: String = "NoamSchedule"
  ): NoamSchedule = {
    new NoamSchedule(warmUpSteps, hiddenSize)
  }
}
