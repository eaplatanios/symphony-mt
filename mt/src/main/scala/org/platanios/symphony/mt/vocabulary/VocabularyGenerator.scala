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

package org.platanios.symphony.mt.vocabulary

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.utilities.MutableFile

import better.files.File

/** Vocabulary creator.
  *
  * Given a sequence of tokenized (i.e., words separated by spaces) text files, vocabulary generators can be used to
  * generated vocabulary files.
  *
  * @author Emmanouil Antonios Platanios
  */
trait VocabularyGenerator {
  /** Returns the vocabulary file name that this generator uses / will use.
    *
    * @param  language Language for which a vocabulary will be generated.
    * @return Vocabulary file name.
    */
  def filename(language: Language): String = s"vocab.${language.abbreviation}"

  /** Generates/Replaces a vocabulary file given a sequence of tokenized text files.
    *
    * @param  language       Language for which a vocabulary will be generated.
    * @param  tokenizedFiles Tokenized text files to use for generating the vocabulary file.
    * @param  vocabDir       Directory in which to save the generated vocabulary file.
    * @return The generated/replaced vocabulary file.
    */
  def generate(language: Language, tokenizedFiles: Seq[MutableFile], vocabDir: File): File

  /** Returns a vocabulary for the specified language, ready to be used by machine translation models.
    *
    * @param  language Language for which to return a vocabulary.
    * @param  vocabDir Directory in which the generated vocabulary file and any other relevant files have been saved.
    * @return Created vocabulary.
    */
  def getVocabulary(language: Language, vocabDir: File): Vocabulary = {
    Vocabulary(vocabDir / filename(language))
  }
}
