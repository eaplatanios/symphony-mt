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
case class WordEmbeddingsPerLanguagePair(embeddingsSize: Int) extends WordEmbeddingsType {
  override type T = Seq[WordEmbeddingsPerLanguagePair.EmbeddingsPair]

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
    val languagePairs = languages
        .combinations(2)
        .map(c => (c(0), c(1)))
        .flatMap(p => Seq(p, (p._2, p._1)))
        .toSeq
    languagePairs.map(pair => tf.variableScope(s"${pair._1._1.abbreviation}-${pair._2._1.abbreviation}") {
      WordEmbeddingsPerLanguagePair.EmbeddingsPair(
        embeddings1 = tf.variable[Float](
          name = pair._1._1.name,
          shape = Shape(pair._1._2.size, embeddingsSize),
          initializer = tf.RandomUniformInitializer(-0.1f, 0.1f)).value,
        embeddings2 = tf.variable[Float](
          name = pair._2._1.name,
          shape = Shape(pair._2._2.size, embeddingsSize),
          initializer = tf.RandomUniformInitializer(-0.1f, 0.1f)).value)
    })
  }

  override def lookupTable(
      lookupTable: Output[Resource],
      languageId: Output[Int]
  ): Output[Resource] = {
    lookupTable.gather(languageId)
  }

  override def embeddingLookup(
      embeddingTables: Seq[WordEmbeddingsPerLanguagePair.EmbeddingsPair],
      languageIds: Seq[Output[Int]],
      languageId: Output[Int],
      keys: Output[Int]
  )(implicit context: Output[Int]): Output[Float] = {
    val languageIdPairs = languageIds
        .combinations(2)
        .map(c => (c(0), c(1)))
        .flatMap(p => Seq(p, (p._2, p._1)))
        .toSeq
    val predicates = embeddingTables.zip(languageIdPairs).flatMap {
      case (embeddings, (srcLangId, tgtLangId)) =>
        val pairPredicate = tf.logicalAnd(
          tf.equal(context(0), srcLangId),
          tf.equal(context(1), tgtLangId))
        Seq(
          (tf.logicalAnd(pairPredicate, tf.equal(srcLangId, languageId)), () => embeddings.embeddings1),
          (tf.logicalAnd(pairPredicate, tf.equal(tgtLangId, languageId)), () => embeddings.embeddings2))
    }
    val assertion = tf.assert(
      tf.any(tf.stack(predicates.map(_._1))),
      Seq(
        tf.constant[String]("No word embeddings table found for the provided language pair."),
        tf.constant[String]("Context source language: "), context(0),
        tf.constant[String]("Context target language: "), context(1),
        tf.constant[String]("Current language: "), languageId))
    val default = () => tf.createWith(controlDependencies = Set(assertion)) {
      tf.identity(embeddingTables.head.embeddings1)
    }
    tf.cases(predicates, default).gather(keys)
  }

  override def projectionToWords(
      languages: Seq[(Language, Vocabulary)],
      languageIds: Seq[Output[Int]],
      projectionsToWords: mutable.Map[Int, T],
      inputSize: Int,
      languageId: Output[Int]
  )(implicit context: Output[Int]): Output[Float] = {
    val projectionsForSize = projectionsToWords
        .getOrElseUpdate(inputSize, {
          val languagePairs = languages
              .combinations(2)
              .map(c => (c(0), c(1)))
              .flatMap(p => Seq(p, (p._2, p._1)))
              .toSeq
          languagePairs.map(pair => tf.variableScope(s"${pair._1._1.abbreviation}-${pair._2._1.abbreviation}") {
            WordEmbeddingsPerLanguagePair.EmbeddingsPair(
              embeddings1 = tf.variable[Float](
                name = s"${pair._1._1.name}/OutWeights",
                shape = Shape(inputSize, pair._1._2.size),
                initializer = tf.RandomUniformInitializer(-0.1f, 0.1f)).value,
              embeddings2 = tf.variable[Float](
                name = s"${pair._2._1.name}/OutWeights",
                shape = Shape(inputSize, pair._2._2.size),
                initializer = tf.RandomUniformInitializer(-0.1f, 0.1f)).value)
          })
        })
    val languageIdPairs = languageIds
        .combinations(2)
        .map(c => (c(0), c(1)))
        .flatMap(p => Seq(p, (p._2, p._1)))
        .toSeq
    val predicates = projectionsForSize.zip(languageIdPairs).flatMap {
      case (projections, (srcLangId, tgtLangId)) =>
        val pairPredicate = tf.logicalAnd(
          tf.equal(context(0), srcLangId),
          tf.equal(context(1), tgtLangId))
        Seq(
          (tf.logicalAnd(pairPredicate, tf.equal(srcLangId, languageId)), () => projections.embeddings1),
          (tf.logicalAnd(pairPredicate, tf.equal(tgtLangId, languageId)), () => projections.embeddings2))
    }
    val assertion = tf.assert(
      tf.any(tf.stack(predicates.map(_._1))),
      Seq(
        tf.constant[String]("No projections found for the provided language pair."),
        tf.constant[String]("Context source language: "), context(0),
        tf.constant[String]("Context target language: "), context(1),
        tf.constant[String]("Current language: "), languageId))
    val default = () => tf.createWith(controlDependencies = Set(assertion)) {
      tf.identity(projectionsForSize.head.embeddings1)
    }
    tf.cases(predicates, default)
  }
}

object WordEmbeddingsPerLanguagePair {
  case class EmbeddingsPair(
      embeddings1: Output[Float],
      embeddings2: Output[Float])

  private[WordEmbeddingsPerLanguagePair] def languagePairIdsToId(
      numLanguages: Int,
      srcLanguageId: Output[Int],
      tgtLanguageId: Output[Int]
  ): Output[Int] = {
    srcLanguageId * (numLanguages - 1) + tgtLanguageId -
        tf.less(srcLanguageId, tgtLanguageId).toInt
  }
}
