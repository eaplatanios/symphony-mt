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

/** Vocabulary generator.
  *
  * Given a sequence of tokenized (i.e., words separated by spaces) text files, vocabulary generators can be used to
  * generate vocabulary files.
  *
  * @author Emmanouil Antonios Platanios
  */
trait VocabularyGenerator {
  /** Returns the vocabulary filename that this generator uses / will use.
    *
    * @param  languages Languages for which a vocabulary will be generated.
    * @return Vocabulary filename.
    */
  def filename(languages: Seq[Language]): String = {
    s"vocab.${languages.map(_.abbreviation).sorted.mkString(".")}"
  }

  /** Generates/Replaces a vocabulary file given a sequence of tokenized text files.
    *
    * @param  languages      Languages for which a merged vocabulary will be generated.
    * @param  tokenizedFiles Tokenized text files to use for generating the vocabulary file.
    * @param  vocabDir       Directory in which to save the generated vocabulary files.
    * @return The generated/replaced vocabulary file.
    */
  protected def generate(
      languages: Seq[Language],
      tokenizedFiles: Seq[MutableFile],
      vocabDir: File
  ): File

  /** Generates/Replaces the vocabulary files for multiple languages, given a sequence of tokenized text files per
    * language.
    *
    * @param  languages      Languages for which a vocabulary will be generated.
    * @param  tokenizedFiles Tokenized text files to use for generating the vocabulary files, for each language
    * @param  vocabDir       Directory in which to save the generated vocabulary files.
    * @param  shared         If `true`, a single vocabulary will be created and shared for all languages.
    * @return The generated/replaced vocabulary files.
    */
  final def generate(
      languages: Seq[Language],
      tokenizedFiles: Seq[Seq[MutableFile]],
      vocabDir: File,
      shared: Boolean = false
  ): Seq[File] = {
    if (shared) {
      val files = tokenizedFiles.flatten
      val vocabulary = generate(languages, files, vocabDir)
      languages.map(_ => vocabulary)
    } else {
      languages.zip(tokenizedFiles).map(a => {
        val languages = Seq(a._1)
        generate(languages, a._2, vocabDir)
      })
    }
  }

  /** Returns a vocabulary for the specified languages, ready to be used by machine translation models.
    *
    * @param  languages Languages for which to return a vocabulary.
    * @param  vocabDir  Directory in which the generated vocabulary file and any other relevant files have been saved.
    * @return Created vocabulary.
    */
  protected def getVocabulary(languages: Seq[Language], vocabDir: File): Vocabulary = {
    Vocabulary(vocabDir / filename(languages))
  }

  /** Returns vocabularies for the specified languages, ready to be used by machine translation models.
    *
    * @param  languages Languages for which to return vocabularies.
    * @param  vocabDir  Directory in which the generated vocabulary files and any other relevant files have been saved.
    * @param  shared    If `true`, a single vocabulary will be created and shared for all languages.
    * @return Created vocabularies.
    */
  final def getVocabularies(
      languages: Seq[Language],
      vocabDir: File,
      shared: Boolean = false
  ): Seq[Vocabulary] = {
    if (shared) {
      val vocabulary = getVocabulary(languages, vocabDir)
      languages.map(_ => vocabulary)
    } else {
      languages.map(l => getVocabulary(Seq(l), vocabDir))
    }
  }

  override def toString: String
}
