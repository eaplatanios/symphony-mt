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

import better.files._

/**
  * @author Emmanouil Antonios Platanios
  */
trait FileProcessor {
  def apply(file: File, language: Language): File = process(file, language)

  def apply(file1: File, file2: File, language1: Language, language2: Language): (File, File) = {
    processPair(file1, file2, language1, language2)
  }

  def process(file: File, language: Language): File = file

  def processPair(file1: File, file2: File, language1: Language, language2: Language): (File, File) = {
    (process(file1, language1), process(file2, language2))
  }

  def >>(processor: FileProcessor): ComposedFileProcessor = {
    ComposedFileProcessor(this, processor)
  }
}

object NoFileProcessor extends FileProcessor

case class ComposedFileProcessor(fileProcessors: FileProcessor*) extends FileProcessor {
  override def process(file: File, language: Language): File = {
    fileProcessors.foldLeft(file)((f, p) => p.process(f, language))
  }

  override def processPair(file1: File, file2: File, language1: Language, language2: Language): (File, File) = {
    fileProcessors.foldLeft((file1, file2))((f, p) => p.processPair(f._1, f._2, language1, language2))
  }
}
