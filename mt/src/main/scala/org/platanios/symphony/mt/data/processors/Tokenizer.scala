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
trait Tokenizer extends FileProcessor {
  override def process(file: File, language: Language): File = {
    tokenizeCorpus(file, language)
  }

  def tokenizedFile(originalFile: File): File
  def tokenize(sentence: String, language: Language): String

  def tokenizeCorpus(file: File, language: Language, bufferSize: Int = 8192): File = {
    val tokenized = tokenizedFile(file)
    if (tokenized.notExists) {
      Tokenizer.logger.info(s"Tokenizing '$file'.")
      val tokenizedWriter = newWriter(tokenized)
      newReader(file).lines().toAutoClosedIterator.foreach(sentence => {
        tokenizedWriter.write(s"${tokenize(sentence, language)}\n")
      })
      tokenizedWriter.flush()
      tokenizedWriter.close()
      Tokenizer.logger.info(s"Created tokenized file '$tokenized'.")
    }
    tokenized
  }

  override def toString: String
}

object Tokenizer {
  private[data] val logger = Logger(LoggerFactory.getLogger("Data / Tokenizer"))
}
