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
import org.platanios.symphony.mt.config.TrainingConfig
import org.platanios.symphony.mt.models.ModelConstructionContext
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
case class SharedWordEmbeddings(embeddingsSize: Int) extends WordEmbeddingsType {
  override type T = Variable[Float]

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
      languages: Seq[(Language, Vocabulary)],
      trainingConfig: TrainingConfig
  ): Variable[Float] = {
    val someLanguage = languages.head
    tf.variable[Float](
      name = languages.map(_._1.name).mkString(""),
      shape = Shape(someLanguage._2.size, embeddingsSize),
      initializer = tf.RandomUniformInitializer(-0.1f, 0.1f))
  }

  override def lookupTable(
      lookupTable: Output[Resource],
      languageId: Output[Int]
  ): Output[Resource] = {
    lookupTable
  }

  override def embeddingsTable(
      embeddingTables: Variable[Float],
      languageIds: Seq[Output[Int]],
      languageId: Output[Int]
  )(implicit context: ModelConstructionContext): Output[Float] = {
    embeddingTables
  }

  override def embeddings(
      embeddingTables: Variable[Float],
      languageIds: Seq[Output[Int]],
      languageId: Output[Int],
      tokenIndices: Output[Int]
  )(implicit context: ModelConstructionContext): Output[Float] = {
    embeddingTables.gather(tokenIndices)
  }

  override def projectionToWords(
      languageIds: Seq[Output[Int]],
      projectionsToWords: mutable.Map[Int, Variable[Float]],
      inputSize: Int,
      languageId: Output[Int]
  )(implicit context: ModelConstructionContext): Output[Float] = {
    projectionsToWords
        .getOrElseUpdate(inputSize, {
          val weightsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
          val someLanguage = context.languages.head
          tf.variable[Float](
            s"${someLanguage._1.name}/OutWeights",
            Shape(inputSize, someLanguage._2.size), weightsInitializer)
        }).value
  }
}
