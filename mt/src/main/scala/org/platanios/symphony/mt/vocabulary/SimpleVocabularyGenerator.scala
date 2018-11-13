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
import org.platanios.symphony.mt.data.{newReader, newWriter}
import org.platanios.symphony.mt.utilities.{MutableFile, TrieWordCounter}

import better.files._
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.util.matching.Regex

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
  protected val whitespaceRegex: Regex = "\\s+".r

  /** Returns the vocabulary file name that this generator uses / will use.
    *
    * @param  languages Languages for which a vocabulary will be generated.
    * @return Vocabulary file name.
    */
  override def filename(languages: Seq[Language]): String = {
    val suffix = languages.map(_.abbreviation).sorted.mkString(".")
    if (sizeThreshold == -1 && countThreshold == -1)
      s"vocab.$suffix"
    else if (countThreshold == -1)
      s"vocab.s$sizeThreshold.$suffix"
    else if (sizeThreshold == -1)
      s"vocab.c$countThreshold.$suffix"
    else
      s"vocab.s$sizeThreshold.c$countThreshold.$suffix"
  }

  /** Generates/Replaces a vocabulary file given a sequence of tokenized text files.
    *
    * @param  languages      Languages for which a merged vocabulary will be generated.
    * @param  tokenizedFiles Tokenized text files to use for generating the vocabulary file.
    * @param  vocabDir       Directory in which to save the generated vocabulary files.
    * @return The generated/replaced vocabulary file.
    */
  override protected def generate(
      languages: Seq[Language],
      tokenizedFiles: Seq[MutableFile],
      vocabDir: File
  ): File = {
    val vocabFile = vocabDir / filename(languages)
    if (replaceExisting || vocabFile.notExists) {
      SimpleVocabularyGenerator.logger.info(s"Generating vocabulary file for ${languages.mkString(", ")}.")
      vocabFile.parent.createDirectories()
      val writer = newWriter(vocabFile)
      tokenizedFiles.map(_.get).toIterator.flatMap(file => {
        newReader(file).lines().toAutoClosedIterator
            .flatMap(whitespaceRegex.split)
      }).foldLeft(TrieWordCounter())((counter, word) => {
        counter.insertWord(word.trim)
        counter
      }).words(sizeThreshold, countThreshold)
          .toSeq
          .sortBy(-_._1)
          .map(_._2)
          .distinct
          .foreach(word => writer.write(word + "\n"))
      writer.flush()
      writer.close()
      SimpleVocabularyGenerator.logger.info(s"Generated vocabulary file for ${languages.mkString(", ")}.")
    } else {
      SimpleVocabularyGenerator.logger.info(s"Vocabulary for ${languages.mkString(", ")} already exists: $vocabFile.")
    }
    vocabFile
  }

  override def toString: String = s"simple-$sizeThreshold-$countThreshold"
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
