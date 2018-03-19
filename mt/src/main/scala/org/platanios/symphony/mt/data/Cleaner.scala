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

import better.files._
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.charset.StandardCharsets

import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
trait Cleaner {
  def cleanFile(originalFile: File): File = {
    val fileName = originalFile.nameWithoutExtension(includeAll = false) + s".clean${originalFile.extension().get}"
    originalFile.sibling(fileName)
  }

  def processPair(srcSentence: String, tgtSentence: String): Option[(String, String)]

  def processCorporaPair(srcFile: File, tgtFile: File, bufferSize: Int = 8192): (File, File) = {
    val srcClean = cleanFile(srcFile)
    val tgtClean = cleanFile(tgtFile)
    if (srcClean.notExists || tgtClean.notExists) {
      Cleaner.logger.info(s"Cleaning '$srcFile' and '$tgtFile'.")
      val srcWriter = newWriter(srcClean)
      val tgtWriter = newWriter(tgtClean)
      newReader(srcFile).lines().toAutoClosedIterator
          .zip(newReader(tgtFile).lines().toAutoClosedIterator).foreach(pair => {
        processPair(pair._1, pair._2) match {
          case Some((srcSentence, tgtSentence)) if srcSentence.length > 0 && tgtSentence.length > 0 =>
            srcWriter.write(s"$srcSentence\n")
            tgtWriter.write(s"$tgtSentence\n")
          case None => ()
        }
      })
      srcWriter.flush()
      srcWriter.close()
      tgtWriter.flush()
      tgtWriter.close()
      Cleaner.logger.info(s"Created clean files '$srcClean' and '$tgtClean'.")
    }
    (srcClean, tgtClean)
  }
}

object Cleaner {
  private[data] val logger = Logger(LoggerFactory.getLogger("Data / Cleaner"))
}

object NoCleaner extends Cleaner {
  override def cleanFile(originalFile: File): File = originalFile

  override def processPair(srcSentence: String, tgtSentence: String): Option[(String, String)] = {
    Some((srcSentence, tgtSentence))
  }

  override def processCorporaPair(srcFile: File, tgtFile: File, bufferSize: Int = 8192): (File, File) = {
    (srcFile, tgtFile)
  }
}

class MosesCleaner protected (
    val minSentenceLength: Int = -1,
    val maxSentenceLength: Int = -1,
    val maxWordLength: Int = -1,
    val lowerCase: Boolean = false
) extends Cleaner {
  protected val ignoredRegex      : Regex = """\|""".r
  protected val whitespaceRegex   : Regex = """\s+""".r
  protected val maxWordLengthRegex: Regex = s"""[\\S]{${maxWordLength + 1},}""".r

  override def processPair(srcSentence: String, tgtSentence: String): Option[(String, String)] = {
    var src = srcSentence.trim
    var tgt = tgtSentence.trim

    // Fix spaces.
    src = whitespaceRegex.replaceAllIn(src, " ")
    tgt = whitespaceRegex.replaceAllIn(tgt, " ")

    // Decide whether to keep the pair or not.
    var keep = src != "" && tgt != ""
    if (minSentenceLength >= 0 || maxSentenceLength >= 0) {
      val srcNumWords = whitespaceRegex.split(src).length
      val tgtNumWords = whitespaceRegex.split(tgt).length
      keep &&= minSentenceLength < 0 || (srcNumWords >= minSentenceLength && tgtNumWords >= minSentenceLength)
      keep &&= maxSentenceLength < 0 || (srcNumWords <= maxSentenceLength && tgtNumWords <= maxSentenceLength)
    }

    // Check for the maximum word length, if necessary.
    if (maxWordLength >= 0) {
      keep &&= maxWordLengthRegex.findFirstIn(src).isEmpty
      keep &&= maxWordLengthRegex.findFirstIn(tgt).isEmpty
    }

    if (!keep) {
      None
    } else {
      // Lowercase, if necessary.
      if (lowerCase) {
        src = src.toLowerCase
        tgt = tgt.toLowerCase
      }

      // Remove non-UTF8 characters.
      src = StandardCharsets.UTF_8.decode(StandardCharsets.UTF_8.encode(src)).toString
      tgt = StandardCharsets.UTF_8.decode(StandardCharsets.UTF_8.encode(tgt)).toString

      Some((src, tgt))
    }
  }
}

object MosesCleaner {
  def apply(
      minSentenceLength: Int = -1,
      maxSentenceLength: Int = -1,
      maxWordLength: Int = -1,
      lowerCase: Boolean = false
  ): MosesCleaner = {
    new MosesCleaner(
      minSentenceLength,
      maxSentenceLength,
      maxWordLength,
      lowerCase)
  }
}
