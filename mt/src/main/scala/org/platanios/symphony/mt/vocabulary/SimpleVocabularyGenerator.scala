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
import org.platanios.symphony.mt.utilities.{MutableFile, TrieWordCounter}

import better.files.File
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.io.BufferedWriter
import java.nio.charset.StandardCharsets
import java.nio.file.StandardOpenOption

import scala.io.Source

/** Simple vocabulary generator that generates a vocabulary by simply splitting the input text files on spaces.
  *
  * @param  sizeThreshold   Vocabulary size threshold. If non-negative, then the created vocabulary size will be
  *                         bounded by this number. This means that only the `sizeThreshold` most frequent words will
  *                         be kept.
  * @param  countThreshold  Vocabulary count threshold. If non-negative, then all words with counts less than
  *                         `countThreshold` will be ignored.
  * @param  replaceExisting If `true`, existing vocabulary files will be replaced, if found.
  * @param  bufferSize      Buffer size to use while reading and writing files.
  *
  * @author Emmanouil Antonios Platanios
  */
class SimpleVocabularyGenerator protected (
    val sizeThreshold: Int = -1,
    val countThreshold: Int = -1,
    val replaceExisting: Boolean = false,
    val bufferSize: Int = 8192
) extends VocabularyGenerator {
  /** Returns the vocabulary file name that this generator uses / will use.
    *
    * @param  language Language for which a vocabulary will be generated.
    * @return Vocabulary file name.
    */
  override def filename(language: Language): String = {
    if (sizeThreshold == -1 && countThreshold == -1)
      s"vocab.${language.abbreviation}"
    else if (countThreshold == -1)
      s"vocab.s$sizeThreshold.${language.abbreviation}"
    else if (sizeThreshold == -1)
      s"vocab.c$countThreshold.${language.abbreviation}"
    else
      s"vocab.s$sizeThreshold.c$countThreshold.${language.abbreviation}"
  }

  /** Generates/Replaces a vocabulary file given a sequence of tokenized text files.
    *
    * @param  language       Language for which a vocabulary will be generated.
    * @param  tokenizedFiles Tokenized text files to use for generating the vocabulary file.
    * @param  vocabDir       Directory in which to save the generated vocabulary file.
    * @return The generated/replaced vocabulary file.
    */
  override def generate(language: Language, tokenizedFiles: Seq[MutableFile], vocabDir: File): File = {
    val vocabFile = vocabDir / filename(language)
    if (replaceExisting || vocabFile.notExists) {
      SimpleVocabularyGenerator.logger.info(s"Generating vocabulary file for $language.")
      vocabFile.parent.createDirectories()
      val whitespaceRegex = "\\s+".r
      val writer = new BufferedWriter(
        vocabFile.newPrintWriter()(Seq(
          StandardOpenOption.CREATE,
          StandardOpenOption.WRITE,
          StandardOpenOption.TRUNCATE_EXISTING)), bufferSize)
      tokenizedFiles.map(_.get).toIterator.flatMap(file => {
        Source.fromFile(file.toJava)(StandardCharsets.UTF_8)
            .getLines
            .flatMap(whitespaceRegex.split)
      }).foldLeft(TrieWordCounter())((counter, word) => {
        counter.insertWord(word)
        counter
      }).words(sizeThreshold, countThreshold)
          .map(_._2).filter(_ != "").toSet[String]
          .foreach(word => writer.write(word + "\n"))
      writer.flush()
      writer.close()
      SimpleVocabularyGenerator.logger.info(s"Generated vocabulary file for $language.")
    } else {
      SimpleVocabularyGenerator.logger.info(s"Vocabulary file for $language already exists: $vocabFile.")
    }
    vocabFile
  }
}

object SimpleVocabularyGenerator {
  private[SimpleVocabularyGenerator] val logger = Logger(LoggerFactory.getLogger("Vocabulary / Simple Generator"))

  def apply(
      sizeThreshold: Int = -1,
      countThreshold: Int = -1,
      replaceExisting: Boolean = false,
      bufferSize: Int = 8192
  ): SimpleVocabularyGenerator = {
    new SimpleVocabularyGenerator(sizeThreshold, countThreshold, replaceExisting, bufferSize)
  }
}
