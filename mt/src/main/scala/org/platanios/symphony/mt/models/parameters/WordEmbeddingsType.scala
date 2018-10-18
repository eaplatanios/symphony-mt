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

package org.platanios.symphony.mt.models.parameters

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.Resource

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
trait WordEmbeddingsType {
  type T

  val embeddingsSize: Int

  def createStringToIndexLookupTable(
      languages: Seq[(Language, Vocabulary)]
  ): Output[Resource]

  def createIndexToStringLookupTable(
      languages: Seq[(Language, Vocabulary)]
  ): Output[Resource]

  def createWordEmbeddings(
      languages: Seq[(Language, Vocabulary)]
  ): T

  def lookupTable(
      lookupTable: Output[Resource],
      languageId: Output[Int]
  ): Output[Resource]

  def embeddingLookup(
      embeddingTables: T,
      languageIds: Seq[Output[Int]],
      languageId: Output[Int],
      keys: Output[Int]
  )(implicit context: Output[Int]): Output[Float]

  def projectionToWords(
      languages: Seq[(Language, Vocabulary)],
      languageIds: Seq[Output[Int]],
      projectionsToWords: mutable.Map[Int, T],
      inputSize: Int,
      languageId: Output[Int]
  )(implicit context: Output[Int]): Output[Float]
}
