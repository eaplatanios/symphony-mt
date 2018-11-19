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
import org.platanios.symphony.mt.data.{newReader, newWriter}

import better.files._
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
object SGMConverter extends FileProcessor {
  private val logger = Logger(LoggerFactory.getLogger("Data / SGM Converter"))

  private val startRegex           : Regex = """(?i)<seg[^>]+>\s*$""".r
  private val startCaptureRegex    : Regex = """(?i)<seg[^>]+>\s*(.*)\s*$""".r
  private val startStopCaptureRegex: Regex = """(?i)<seg[^>]+>\s*(.*)\s*<\/seg>""".r
  private val whitespaceRegex      : Regex = """\s+""".r

  override def process(file: File, language: Language): File = {
    val textFile = convertedFile(file)
    if (textFile.notExists) {
      logger.info(s"Converting SGM file '$file' to text file '$textFile'.")
      val reader = newReader(file)
      val writer = newWriter(textFile)
      val linesIterator = reader.lines().iterator().asScala
      while (linesIterator.nonEmpty) {
        var line = linesIterator.next()
        while (linesIterator.hasNext && startRegex.findFirstIn(line).isDefined)
          line += linesIterator.next()
        while (linesIterator.hasNext &&
            startCaptureRegex.findFirstIn(line).isDefined &&
            startStopCaptureRegex.findFirstIn(line).isEmpty)
          line += linesIterator.next()
        startStopCaptureRegex.findFirstMatchIn(line) match {
          case Some(sentenceMatch) =>
            var sentence = sentenceMatch.group(1)
            sentence = whitespaceRegex.replaceAllIn(sentence.trim, " ")
            writer.write(s"$sentence\n")
          case None => ()
        }
      }
      writer.flush()
      writer.close()
      logger.info(s"Converted SGM file '$file' to text file '$textFile'.")
    }
    textFile
  }

  private def convertedFile(originalFile: File): File = {
    originalFile.sibling(originalFile.nameWithoutExtension(includeAll = false))
  }
}
