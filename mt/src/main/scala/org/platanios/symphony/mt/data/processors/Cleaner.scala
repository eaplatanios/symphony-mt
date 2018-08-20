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

/**
  * @author Emmanouil Antonios Platanios
  */
trait Cleaner extends FileProcessor {
  override def apply(file1: File, file2: File, language1: Language, language2: Language): (File, File) = {
    cleanCorporaPair(file1, file2)
  }

  def cleanFile(originalFile: File): File
  def cleanSentencePair(srcSentence: String, tgtSentence: String): Option[(String, String)]

  def cleanCorporaPair(srcFile: File, tgtFile: File, bufferSize: Int = 8192): (File, File) = {
    val srcClean = cleanFile(srcFile)
    val tgtClean = cleanFile(tgtFile)
    if (srcClean.notExists || tgtClean.notExists) {
      Cleaner.logger.info(s"Cleaning '$srcFile' and '$tgtFile'.")
      val srcWriter = newWriter(srcClean)
      val tgtWriter = newWriter(tgtClean)
      newReader(srcFile).lines().toAutoClosedIterator
          .zip(newReader(tgtFile).lines().toAutoClosedIterator).foreach(pair => {
        cleanSentencePair(pair._1, pair._2) match {
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

  override def toString: String
}

object Cleaner {
  private[data] val logger = Logger(LoggerFactory.getLogger("Data / Cleaner"))
}
