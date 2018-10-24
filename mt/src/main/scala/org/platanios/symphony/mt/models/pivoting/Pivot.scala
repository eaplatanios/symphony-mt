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

package org.platanios.symphony.mt.models.pivoting

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.models.parameters.ParameterManager
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._

/**
  * @author Emmanouil Antonios Platanios
  */
trait Pivot {
  def initialize(
      languages: Seq[(Language, Vocabulary)],
      parameterManager: ParameterManager
  ): Unit

  def pivotingSequence(
      srcLanguage: Output[Int],
      tgtLanguage: Output[Int]
  ): Output[Int]
}
