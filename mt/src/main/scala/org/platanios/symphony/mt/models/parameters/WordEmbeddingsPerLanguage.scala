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
case class WordEmbeddingsPerLanguage(embeddingsSize: Int) extends WordEmbeddingsType {
  override type T = Seq[Variable[Float]]

  override def createStringToIndexLookupTable(
      languages: Seq[(Language, Vocabulary)]
  ): Output[Resource] = {
    val tables = languages.map(l => l._2.stringToIndexLookupTable(name = l._1.name))
    tf.stack(tables.map(_.handle))
  }

  override def createIndexToStringLookupTable(
      languages: Seq[(Language, Vocabulary)]
  ): Output[Resource] = {
    val tables = languages.map(l => l._2.indexToStringLookupTable(name = l._1.name))
    tf.stack(tables.map(_.handle))
  }

  override def createWordEmbeddings(
      languages: Seq[(Language, Vocabulary)],
      trainingConfig: TrainingConfig
  ): T = {
    val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
    languages.map(l => {
      tf.variable[Float](l._1.name, Shape(l._2.size, embeddingsSize), embeddingsInitializer)
    })
  }

  override def lookupTable(
      lookupTable: Output[Resource],
      languageId: Output[Int]
  ): Output[Resource] = {
    lookupTable.gather(languageId)
  }

  override def embeddingsTable(
      embeddingTables: Seq[Variable[Float]],
      languageIds: Seq[Output[Int]],
      languageId: Output[Int]
  )(implicit context: ModelConstructionContext): Output[Float] = {
    val predicates = embeddingTables.zip(languageIds).map {
      case (embeddings, langId) => (tf.equal(languageId, langId), () => embeddings.value)
    }
    val assertion = tf.assert(
      tf.any(tf.stack(predicates.map(_._1))),
      Seq[Output[Any]](
        tf.constant[String]("No word embeddings table found for the provided language."),
        tf.constant[String]("Current language: "), languageId))
    val default = () => tf.createWith(controlDependencies = Set(assertion)) {
      tf.identity(embeddingTables.head.value)
    }
    tf.cases(predicates, default)
  }

  override def embeddings(
      embeddingTables: Seq[Variable[Float]],
      languageIds: Seq[Output[Int]],
      languageId: Output[Int],
      tokenIndices: Output[Int]
  )(implicit context: ModelConstructionContext): Output[Float] = {
    val predicates = embeddingTables.zip(languageIds).map {
      case (embeddings, langId) => (tf.equal(languageId, langId), () => embeddings.gather(tokenIndices))
    }
    val assertion = tf.assert(
      tf.any(tf.stack(predicates.map(_._1))),
      Seq[Output[Any]](
        tf.constant[String]("No word embeddings table found for the provided language."),
        tf.constant[String]("Current language: "), languageId))
    val default = () => tf.createWith(controlDependencies = Set(assertion)) {
      tf.identity(embeddingTables.head.gather(tokenIndices))
    }
    tf.cases(predicates, default)
  }

  override def projectionToWords(
      languageIds: Seq[Output[Int]],
      projectionsToWords: mutable.Map[Int, Seq[Variable[Float]]],
      inputSize: Int,
      languageId: Output[Int]
  )(implicit context: ModelConstructionContext): Output[Float] = {
    val projectionsForSize = projectionsToWords
        .getOrElseUpdate(inputSize, {
          context.languages.map(l => {
            tf.variable[Float](
              name = s"${l._1.name}/OutWeights",
              shape = Shape(inputSize, l._2.size),
              initializer = tf.RandomUniformInitializer(-0.1f, 0.1f))
          })
        })
    val predicates = projectionsForSize.zip(languageIds).map {
      case (projections, langId) => (tf.equal(languageId, langId), () => projections.value)
    }
    val assertion = tf.assert(
      tf.any(tf.stack(predicates.map(_._1))),
      Seq[Output[Any]](
        tf.constant[String]("No projections found for the provided language."),
        tf.constant[String]("Current language: "),
        languageId))
    val default = () => tf.createWith(controlDependencies = Set(assertion)) {
      tf.identity(projectionsForSize.head.value)
    }
    tf.cases(predicates, default)
  }
}
