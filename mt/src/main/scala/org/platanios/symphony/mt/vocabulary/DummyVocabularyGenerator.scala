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

import java.io.{BufferedWriter, OutputStreamWriter}

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.utilities.MutableFile

import better.files._
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.charset.StandardCharsets
import java.nio.file.StandardOpenOption

/** Dummy vocabulary generator that generates a vocabulary containing numbers starting at `0` and ending at `size - 1`.
  *
  * @param  size            Vocabulary size to use.
  * @param  replaceExisting If `true`, existing vocabulary files will be replaced, if found.
  * @param  bufferSize      Buffer size to use while writing vocabulary files.
  *
  * @author Emmanouil Antonios Platanios
  */
class DummyVocabularyGenerator protected (
    val size: Int,
    val replaceExisting: Boolean = false,
    val bufferSize: Int = 8192
) extends VocabularyGenerator {
  /** Returns the vocabulary file name that this generator uses / will use.
    *
    * @param  language Language for which a vocabulary will be generated.
    * @return Vocabulary file name.
    */
  override def filename(language: Language): String = {
    s"vocab.dummy.$size.${language.abbreviation}"
  }

  /** Generates/Replaces a vocabulary file given a sequence of tokenized text files.
    *
    * @param  language       Language for which a vocabulary will be generated.
    * @param  tokenizedFiles Tokenized text files to use for generating the vocabulary file.
    * @param  vocabDir       Directory in which to save the generated vocabulary file.
    * @return The generated/replaced vocabulary file.
    */
  def generate(language: Language, tokenizedFiles: Seq[MutableFile], vocabDir: File): File = {
    val vocabFile = vocabDir / filename(language)
    if (replaceExisting || vocabFile.notExists) {
      DummyVocabularyGenerator.logger.info(s"Generating vocabulary file for $language.")
      vocabFile.parent.createDirectories()
      val writer = new BufferedWriter(
        new OutputStreamWriter(
          vocabFile.newOutputStream(Seq(
            StandardOpenOption.CREATE,
            StandardOpenOption.WRITE,
            StandardOpenOption.TRUNCATE_EXISTING)),
          StandardCharsets.UTF_8))
      (0 until size).foreach(wordId => writer.write(wordId + "\n"))
      writer.flush()
      writer.close()
      DummyVocabularyGenerator.logger.info(s"Generated vocabulary file for $language.")
    } else {
      DummyVocabularyGenerator.logger.info(s"Vocabulary file for $language already exists: $vocabFile.")
    }
    vocabFile
  }
}

object DummyVocabularyGenerator {
  private[DummyVocabularyGenerator] val logger = Logger(LoggerFactory.getLogger("Vocabulary / Dummy Generator"))

  def apply(size: Int, replaceExisting: Boolean = false, bufferSize: Int = 8192): DummyVocabularyGenerator = {
    new DummyVocabularyGenerator(size, replaceExisting, bufferSize)
  }
}
