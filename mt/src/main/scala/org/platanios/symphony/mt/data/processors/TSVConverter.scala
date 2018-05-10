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

package org.platanios.symphony.mt.data.processors

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.data._

import better.files.File
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._

/**
  * @author Emmanouil Antonios Platanios
  */
object TSVConverter extends FileProcessor {
  private val logger = Logger(LoggerFactory.getLogger("Data / TSV Converter"))

  private val MISSING_SENTENCE: String = "__NULL__"

  override def processPair(file1: File, file2: File, language1: Language, language2: Language): (File, File) = {
    assert(file1 == file2, "The TSV converter assumes that the data for all languages is stored in the same file.")
    val reader = newReader(file1)
    val linesIterator = reader.lines().iterator().asScala
    val header = linesIterator.next()
    val languages = header.split('\t').tail.map(Language.fromAbbreviation)
    val language1Index = languages.indexOf(language1) + 1 // We add one to account for the first column ("talk_name").
    val language2Index = languages.indexOf(language2) + 1
    val textFileNamePrefix = s"${file1.nameWithoutExtension}.${language1.abbreviation}-${language2.abbreviation}"
    val textFile1 = file1.sibling(s"$textFileNamePrefix.${language1.abbreviation}")
    val textFile2 = file2.sibling(s"$textFileNamePrefix.${language2.abbreviation}")
    if (textFile1.notExists || textFile2.notExists) {
      logger.info(s"Converting TSV file '$file1' to text files '$textFile1' and '$textFile2'.")
      val writer1 = newWriter(textFile1)
      val writer2 = newWriter(textFile2)
      while (linesIterator.nonEmpty) {
        val line = linesIterator.next()
        val lineParts = line.split('\t')
        val language1Sentence = lineParts(language1Index)
        val language2Sentence = lineParts(language2Index)
        if (language1Sentence != MISSING_SENTENCE && language2Sentence != MISSING_SENTENCE) {
          writer1.write(s"${lineParts(language1Index)}\n")
          writer2.write(s"${lineParts(language2Index)}\n")
        }
      }
      writer1.flush()
      writer2.flush()
      writer1.close()
      writer2.close()
      logger.info(s"Converted TSV file '$file1' to text files '$textFile1' and '$textFile2'.")
    }
    (textFile1, textFile2)
  }
}
