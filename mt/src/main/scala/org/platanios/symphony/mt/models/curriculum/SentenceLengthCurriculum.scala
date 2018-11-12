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

package org.platanios.symphony.mt.models.curriculum

import org.platanios.symphony.mt.models.SentencePairs
import org.platanios.symphony.mt.models.curriculum.competency.Competency
import org.platanios.tensorflow.api._

/**
  * Currently this is very simple and only does the following:
  *
  * Go from 0 to 0.5 linearly for sentences up to length 20, and then continue linearly to 1.0 for sentences up to
  * length 100.
  *
  * @author Emmanouil Antonios Platanios
  */
class SentenceLengthCurriculum[T](
    override val competency: Competency[Output[Float]]
) extends DifficultyBasedCurriculum[SentencePairs[T]](competency) {
  override def difficulty(sample: SentencePairs[T]): Output[Float] = {
    val tgtLength = sample._3._2.toFloat
    val part1 = 0.0f // Length 1 to 10.
    val part2 = (tgtLength - 10.0f) / 30.0f // Length 10 to 10.
    val part3 = 0.5f + tgtLength / 100.0f // Length 20 to 100.
    tf.minimum(tf.maximum(part1, tf.minimum(part2, part3)), 1.0f)
  }
}
