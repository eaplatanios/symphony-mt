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
case class WordEmbeddingsPerLanguage(embeddingsSize: Int) extends WordEmbeddingsType {
  override type T = Seq[Output[Float]]

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
      languages: Seq[(Language, Vocabulary)]
  ): T = {
    val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
    languages.map(l => {
      tf.variable[Float](l._1.name, Shape(l._2.size, embeddingsSize), embeddingsInitializer).value
    })
  }

  override def lookupTable(
      lookupTable: Output[Resource],
      languageId: Output[Int]
  ): Output[Resource] = {
    lookupTable.gather(languageId)
  }

  override def embeddingLookup(
      embeddingTables: Seq[Output[Float]],
      languageIds: Seq[Output[Int]],
      languageId: Output[Int],
      keys: Output[Int]
  )(implicit context: Output[Int]): Output[Float] = {
    val predicates = embeddingTables.zip(languageIds).map {
      case (embeddings, langId) => (tf.equal(languageId, langId), () => embeddings)
    }
    val assertion = tf.assert(
      tf.any(tf.stack(predicates.map(_._1))),
      Seq[Output[Any]](
        tf.constant[String]("No word embeddings table found for the provided language."),
        tf.constant[String]("Current language: "),
        languageId))
    val default = () => tf.createWith(controlDependencies = Set(assertion)) {
      tf.identity(embeddingTables.head)
    }
    tf.cases(predicates, default).gather(keys)
  }

  override def projectionToWords(
      languages: Seq[(Language, Vocabulary)],
      languageIds: Seq[Output[Int]],
      projectionsToWords: mutable.Map[Int, Seq[Output[Float]]],
      inputSize: Int,
      languageId: Output[Int]
  )(implicit context: Output[Int]): Output[Float] = {
    val projectionsForSize = projectionsToWords
        .getOrElseUpdate(inputSize, {
          languages.map(l => {
            tf.variable[Float](
              name = s"${l._1.name}/OutWeights",
              shape = Shape(inputSize, l._2.size),
              initializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
            ).value
          })
        })
    val predicates = projectionsForSize.zip(languageIds).map {
      case (projections, langId) => (tf.equal(languageId, langId), () => projections)
    }
    val assertion = tf.assert(
      tf.any(tf.stack(predicates.map(_._1))),
      Seq[Output[Any]](
        tf.constant[String]("No projections found for the provided language."),
        tf.constant[String]("Current language: "),
        languageId))
    val default = () => tf.createWith(controlDependencies = Set(assertion)) {
      tf.identity(projectionsForSize.head)
    }
    tf.cases(predicates, default)
  }
}
