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

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.vocabulary.Vocabulary

/**
  * @author Emmanouil Antonios Platanios
  */
trait ParallelDataset[T <: ParallelDataset[T]] {
  val name      : String
  val vocabulary: Map[Language, Vocabulary]
  val dataConfig: DataConfig

  def supportsLanguage(language: Language): Boolean = vocabulary.contains(language)

  @throws[IllegalArgumentException]
  protected def checkSupportsLanguage(language: Language): Unit = {
    if (!supportsLanguage(language))
      throw new IllegalArgumentException(s"Dataset '$name' does not support the $language language.")
  }

  def languages: Set[Language] = vocabulary.keySet

  def languagePairs: Set[(Language, Language)] = {
    languages.toSeq.combinations(2)
        .map(c => (c(0), c(1)))
        .filter(p => p._1 != p._2)
        .flatMap(p => Seq(p, (p._2, p._1)))
        .toSet
  }

  def filterLanguages(languages: Language*): T
  def filterTypes(fileTypes: DatasetType*): T
  def filterKeys(keys: String*): T

  /** Creates and returns a TensorFlow dataset, for the specified language.
    *
    * Each element of that dataset is a tuple containing:
    *   - `INT32` tensor containing the input sentence word IDs, with shape `[batchSize, maxSentenceLength]`.
    *   - `INT32` tensor containing the input sentence lengths, with shape `[batchSize]`.
    *
    * @param  language   Language for which the TensorFlow dataset is constructed.
    * @param  dataConfig Data configuration to use (optionally overriding this dataset's configuration).
    * @return Created TensorFlow dataset.
    */
  def toTFMonolingual(language: Language, dataConfig: DataConfig = dataConfig): TFMonolingualDataset

  def toTFBilingual(
      language1: Language,
      language2: Language,
      dataConfig: DataConfig = dataConfig,
      repeat: Boolean = true,
      isEval: Boolean = false
  ): TFBilingualDataset
}
