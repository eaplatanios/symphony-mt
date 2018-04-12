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
trait ParallelDataset {
  val name      : String
  val vocabulary: Map[Language, Vocabulary]

  def supportsLanguage(language: Language): Boolean = vocabulary.contains(language)

  @throws[IllegalArgumentException]
  protected def checkSupportsLanguage(language: Language): Unit = {
    if (!supportsLanguage(language))
      throw new IllegalArgumentException(s"Dataset '$name' does not support the $language language.")
  }

  def languages: Set[Language] = vocabulary.keySet

  def languagePairs(
      includeIdentity: Boolean = false,
      includeReverse: Boolean = true
  ): Set[(Language, Language)] = {
    var pairs = languages.toSeq.combinations(2)
        .map(c => (c(0), c(1)))
    if (includeIdentity)
      pairs = pairs.flatMap(p => Seq(p, (p._1, p._1), (p._2, p._2)))
    if (includeReverse)
      pairs = pairs.flatMap(p => Seq(p, (p._2, p._1)))
    pairs.toSet
  }

  def isEmpty: Boolean
  def nonEmpty: Boolean

  def filterLanguages(languages: Language*): ParallelDataset
  def filterTypes(fileTypes: DatasetType*): ParallelDataset
  def filterTags(tags: ParallelDataset.Tag*): ParallelDataset
}

object ParallelDataset {
  trait Tag {
    val value: String
    override def toString: String = value
  }
}
