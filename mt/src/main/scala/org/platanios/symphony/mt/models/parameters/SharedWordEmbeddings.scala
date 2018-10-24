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
case class SharedWordEmbeddings(embeddingsSize: Int) extends WordEmbeddingsType {
  override type T = Output[Float]

  override def createStringToIndexLookupTable(
      languages: Seq[(Language, Vocabulary)]
  ): Output[Resource] = {
    languages.head._2.stringToIndexLookupTable(name = "SharedStringToIndexLookupTable").handle
  }

  override def createIndexToStringLookupTable(
      languages: Seq[(Language, Vocabulary)]
  ): Output[Resource] = {
    languages.head._2.indexToStringLookupTable(name = "SharedIndexToStringLookupTable").handle
  }

  override def createWordEmbeddings(
      languages: Seq[(Language, Vocabulary)]
  ): Output[Float] = {
    val someLanguage = languages.head
    tf.variable[Float](
      name = someLanguage._1.name,
      shape = Shape(someLanguage._2.size, embeddingsSize),
      initializer = tf.RandomUniformInitializer(-0.1f, 0.1f)).value
  }

  override def lookupTable(
      lookupTable: Output[Resource],
      languageId: Output[Int]
  ): Output[Resource] = {
    lookupTable
  }

  override def embeddingLookup(
      embeddingTables: Output[Float],
      languageIds: Seq[Output[Int]],
      languageId: Output[Int],
      keys: Output[Int]
  )(implicit context: Output[Int]): Output[Float] = {
    embeddingTables.gather(keys)
  }

  override def projectionToWords(
      languages: Seq[(Language, Vocabulary)],
      languageIds: Seq[Output[Int]],
      projectionsToWords: mutable.Map[Int, Output[Float]],
      inputSize: Int,
      languageId: Output[Int]
  )(implicit context: Output[Int]): Output[Float] = {
    projectionsToWords
        .getOrElseUpdate(inputSize, {
          val weightsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
          val someLanguage = languages.head
          tf.variable[Float](
            s"${someLanguage._1.name}/OutWeights",
            Shape(inputSize, someLanguage._2.size), weightsInitializer).value
        })
  }
}
