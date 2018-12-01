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

package org.platanios.symphony.mt.data.scores

import org.platanios.symphony.mt.Language

import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
object SentenceLength extends SentenceScore {
  protected val whitespaceRegex: Regex = "\\s+".r

  override def name: String = {
    "sl"
  }

  override def processSentence(
      language: Language,
      sentence: String,
      requiredValues: Seq[Float],
      requiredSummaries: Seq[SummaryScore]
  ): Float = {
    whitespaceRegex.split(sentence).length
  }
}
