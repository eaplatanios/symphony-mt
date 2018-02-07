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

// TODO: Add support for `BPEVocabularyGenerator`.

/** Vocabulary creator.
  *
  * Given a sequence of tokenized (i.e., words separated by spaces) text files, vocabulary generators can be used to
  * generated vocabulary files.
  *
  * @author Emmanouil Antonios Platanios
  */
trait VocabularyGenerator {
  /** Generates/Replaces a vocabulary file given a sequence of tokenized text files.
    *
    * @param  tokenizedFiles Tokenized text files to use for generating the vocabulary file.
    * @param  vocabFile      Vocabulary file to generate/replace.
    * @return The generated/replaced vocabulary file (same as `vocabFile`).
    */
  def generate(tokenizedFiles: Seq[File], vocabFile: File): File
}
