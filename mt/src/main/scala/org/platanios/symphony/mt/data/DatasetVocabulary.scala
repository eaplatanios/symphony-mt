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

package org.platanios.symphony.mt.data

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.vocabulary.VocabularyGenerator

/**
  * @author Emmanouil Antonios Platanios
  */
sealed trait DatasetVocabulary {
  /** Returns the vocabulary file name that this vocabulary uses.
    *
    * @param  languages Languages for which a vocabulary will be generated.
    * @return Vocabulary file name.
    */
  def filename(languages: Seq[Language]): String = s"vocab.${languages.map(_.abbreviation).sorted.mkString(".")}"

  override def toString: String
}

case object NoVocabulary extends DatasetVocabulary {
  override def toString: String = "v:none"
}

case object MergedVocabularies extends DatasetVocabulary {
  override def toString: String = "v:merged"
}

case class GeneratedVocabulary(generator: VocabularyGenerator, shared: Boolean) extends DatasetVocabulary {
  /** Returns the vocabulary file name that this vocabulary uses.
    *
    * @param  languages Languages for which a vocabulary will be generated.
    * @return Vocabulary file name.
    */
  override def filename(languages: Seq[Language]): String = generator.filename(languages)

  override def toString: String = s"v:generated-${generator.toString}"
}
