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

import org.platanios.symphony.mt.utilities.TrieWordCounter

import better.files.File

import java.io.BufferedWriter
import java.nio.charset.StandardCharsets
import java.nio.file.StandardOpenOption

import scala.io.Source

/** Simple vocabulary generator that generates a vocabulary by simply splitting the input text files on spaces.
  *
  * @param  sizeThreshold  Vocabulary size threshold. If non-negative, then the created vocabulary size will be
  *                        bounded by this number. This means that only the `sizeThreshold` most frequent words will
  *                        be kept.
  * @param  countThreshold Vocabulary count threshold. If non-negative, then all words with counts less than
  *                        `countThreshold` will be ignored.
  * @param  bufferSize     Buffer size to use while reading and writing files.
  *
  * @author Emmanouil Antonios Platanios
  */
class SimpleVocabularyGenerator protected (
    val sizeThreshold: Int = -1,
    val countThreshold: Int = -1,
    val bufferSize: Int = 8192
) extends VocabularyGenerator {
  /** Generates/Replaces a vocabulary file given a sequence of tokenized text files.
    *
    * @param  tokenizedFiles Tokenized text files to use for generating the vocabulary file.
    * @param  vocabFile      Vocabulary file to generate/replace.
    * @return The generated/replaced vocabulary file (same as `vocabFile`).
    */
  override def generate(tokenizedFiles: Seq[File], vocabFile: File): File = {
    vocabFile.parent.createDirectories()
    val whitespaceRegex = "\\s+".r
    val writer = new BufferedWriter(
      vocabFile.newPrintWriter()(Seq(
        StandardOpenOption.CREATE,
        StandardOpenOption.WRITE,
        StandardOpenOption.TRUNCATE_EXISTING)), bufferSize)
    tokenizedFiles.toStream.flatMap(file => {
      Source.fromFile(file.toJava)(StandardCharsets.UTF_8)
          .getLines
          .flatMap(whitespaceRegex.split)
    }).foldLeft(TrieWordCounter())((counter, word) => {
      counter.insertWord(word)
      counter
    }).words(sizeThreshold, countThreshold)
        .toSeq.sortBy(-_._1).map(_._2)
        .filter(_ != "")
        .foreach(word => writer.write(word + "\n"))
    writer.flush()
    writer.close()
    vocabFile
  }
}

object SimpleVocabularyGenerator {
  def apply(
      sizeThreshold: Int = -1,
      countThreshold: Int = -1,
      bufferSize: Int = 8192
  ): SimpleVocabularyGenerator = {
    new SimpleVocabularyGenerator(sizeThreshold, countThreshold, bufferSize)
  }
}
