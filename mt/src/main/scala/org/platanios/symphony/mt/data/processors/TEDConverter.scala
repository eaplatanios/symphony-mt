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

import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
object TEDConverter extends FileProcessor {
  private val logger = Logger(LoggerFactory.getLogger("Data / TED Converter"))

  private val ignoredRegex: Regex = """.*(?:<url>|<talkid>|<keywords>|<speaker|<reviewer|<translator).*""".r
  private val removeRegex : Regex = """(?:<title>|</title>|<doc .*>|</doc>|<description>|</description>)""".r

  override def process(file: File, language: Language): File = convertTEDToText(file)

  private def convertedFile(originalFile: File): File = {
    originalFile.sibling("converted." + originalFile.name)
  }

  def convertTEDToText(tedFile: File): File = {
    val textFile = convertedFile(tedFile)
    if (textFile.notExists) {
      logger.info(s"Converting TED file '$tedFile' to text file '$textFile'.")
      val reader = newReader(tedFile)
      val writer = newWriter(textFile)
      reader.lines().toAutoClosedIterator.foreach(line => {
        ignoredRegex.findFirstMatchIn(line) match {
          case Some(_) => ()
          case None => writer.write(s"${removeRegex.replaceAllIn(line, "")}\n")
        }
      })
      writer.flush()
      writer.close()
      logger.info(s"Converted TED file '$tedFile' to text file '$textFile'.")
    }
    textFile
  }
}
