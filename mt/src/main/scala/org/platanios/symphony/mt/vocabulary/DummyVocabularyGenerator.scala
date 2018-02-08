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

import better.files.File

import java.io.BufferedWriter
import java.nio.file.StandardOpenOption

/** Dummy vocabulary generator that generates a vocabulary containing numbers starting at `0` and ending at `size - 1`.
  *
  * @param  size       Vocabulary size to use.
  * @param  bufferSize Buffer size to use while writing vocabulary files.
  *
  * @author Emmanouil Antonios Platanios
  */
class DummyVocabularyGenerator protected (
    val size: Int,
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
    val writer = new BufferedWriter(
      vocabFile.newPrintWriter()(Seq(
        StandardOpenOption.CREATE,
        StandardOpenOption.WRITE,
        StandardOpenOption.TRUNCATE_EXISTING)), bufferSize)
    (0 until size).foreach(wordId => writer.write(wordId + "\n"))
    writer.flush()
    writer.close()
    vocabFile
  }
}

object DummyVocabularyGenerator {
  def apply(size: Int, bufferSize: Int = 8192): DummyVocabularyGenerator = {
    new DummyVocabularyGenerator(size, bufferSize)
  }
}
