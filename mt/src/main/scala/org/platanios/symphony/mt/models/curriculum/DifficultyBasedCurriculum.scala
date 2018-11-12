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

import org.platanios.symphony.mt.models.curriculum.competency.Competency
import org.platanios.tensorflow.api._

// TODO: Things to consider:
//   - Source sentence length
//   - Target sentence length
//   - Length discrepancy
//   - Word frequency
//   - Alignment

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class DifficultyBasedCurriculum[Sample](
    val competency: Competency[Output[Float]]
) extends Curriculum[Sample] {
  def difficulty(sample: Sample): Output[Float]

  override final def samplesFilter: Option[Sample => Output[Boolean]] = {
    Some((sample: Sample) => tf.nameScope("Curriculum/SamplesFilter") {
      tf.lessEqual(difficulty(sample), competency.currentLevel(getCurrentStep))
    })
  }
}
