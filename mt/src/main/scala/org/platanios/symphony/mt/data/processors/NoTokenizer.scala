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

import better.files.File

/** Represents a dummy tokenizer that does nothing.
  *
  * @author Emmanouil Antonios Platanios
  */
object NoTokenizer extends Tokenizer {
  override def tokenizedFile(originalFile: File): File = originalFile
  override def tokenize(sentence: String, language: Language): String = sentence
  override def tokenizeCorpus(file: File, language: Language, bufferSize: Int = 8192): File = file

  override def toString: String = "t:none"
}
