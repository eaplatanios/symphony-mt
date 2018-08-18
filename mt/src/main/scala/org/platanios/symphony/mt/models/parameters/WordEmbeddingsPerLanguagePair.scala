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

import scala.collection.mutable

// TODO: !!! Complete the implementation for the merged versions.

/**
  * @author Emmanouil Antonios Platanios
  */
class WordEmbeddingsPerLanguagePair protected (
    override val embeddingsSize: Int,
    val mergedEmbeddings: Boolean = false,
    val mergedProjections: Boolean = false
) extends WordEmbeddingsType {
  override type T = Seq[(Output, Output)]

  override def createStringToIndexLookupTable(languages: Seq[(Language, Vocabulary)]): Output = {
    val tables = languages.map(l => l._2.stringToIndexLookupTable(name = l._1.name))
    tf.stack(tables.map(_.handle))
  }

  override def createIndexToStringLookupTable(languages: Seq[(Language, Vocabulary)]): Output = {
    val tables = languages.map(l => l._2.indexToStringLookupTable(name = l._1.name))
    tf.stack(tables.map(_.handle))
  }

  override def createWordEmbeddings(languages: Seq[(Language, Vocabulary)]): T = {
    val languagePairs = languages
        .combinations(2)
        .map(c => (c(0), c(1)))
        .flatMap(p => Seq(p, (p._2, p._1)))
        .toSeq
    val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
    if (!mergedEmbeddings) {
      languagePairs.map(pair => tf.variableScope(s"${pair._1._1.abbreviation}-${pair._2._1.abbreviation}") {
        (tf.variable(pair._1._1.name, FLOAT32, Shape(pair._1._2.size, embeddingsSize), embeddingsInitializer).value,
            tf.variable(pair._2._1.name, FLOAT32, Shape(pair._2._2.size, embeddingsSize), embeddingsInitializer).value)
      })
    } else {
      ???
    }
  }

  override def lookupTable(lookupTable: Output, languageId: Output): Output = {
    lookupTable.gather(languageId)
  }

  override def embeddingLookup(
      embeddingTables: T,
      languageIds: Seq[Output],
      languageId: Output,
      keys: Output,
      context: Option[(Output, Output)]
  ): Output = {
    if (!mergedEmbeddings) {
      val languageIdPairs = languageIds
          .combinations(2)
          .map(c => (c(0), c(1)))
          .flatMap(p => Seq(p, (p._2, p._1)))
          .toSeq
      val predicates = embeddingTables.zip(languageIdPairs).flatMap {
        case (embeddings, (srcLangId, tgtLangId)) =>
          val pairPredicate = tf.logicalAnd(
            tf.equal(context.get._1, srcLangId),
            tf.equal(context.get._2, tgtLangId))
          Seq(
            (tf.logicalAnd(pairPredicate, tf.equal(srcLangId, languageId)), () => embeddings._1),
            (tf.logicalAnd(pairPredicate, tf.equal(tgtLangId, languageId)), () => embeddings._2))
      }
      val assertion = tf.assert(
        tf.any(tf.stack(predicates.map(_._1))),
        Seq("No word embeddings table found for the provided language pair."))
      val default = () => tf.createWith(controlDependencies = Set(assertion)) {
        tf.identity(embeddingTables.head._1)
      }
      tf.cases(predicates, default).gather(keys)
    } else {
      ???
    }
  }

  override def projectionToWords(
      languages: Seq[(Language, Vocabulary)],
      languageIds: Seq[Output],
      projectionsToWords: mutable.Map[Int, T],
      inputSize: Int,
      languageId: Output,
      context: Option[(Output, Output)]
  ): Output = {
    if (!mergedProjections) {
      val projectionsForSize = projectionsToWords
          .getOrElseUpdate(inputSize, {
            val languagePairs = languages
                .combinations(2)
                .map(c => (c(0), c(1)))
                .flatMap(p => Seq(p, (p._2, p._1)))
                .toSeq
            val weightsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
            languagePairs.map(pair => tf.variableScope(s"${pair._1._1.abbreviation}-${pair._2._1.abbreviation}") {
              (tf.variable(
                s"${pair._1._1.name}/OutWeights", FLOAT32, Shape(inputSize, pair._1._2.size),
                weightsInitializer).value,
                  tf.variable(
                    s"${pair._2._1.name}/OutWeights", FLOAT32, Shape(inputSize, pair._2._2.size),
                    weightsInitializer).value)
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
            tf.equal(context.get._1, srcLangId),
            tf.equal(context.get._2, tgtLangId))
          Seq(
            (tf.logicalAnd(pairPredicate, tf.equal(srcLangId, languageId)), () => projections._1),
            (tf.logicalAnd(pairPredicate, tf.equal(tgtLangId, languageId)), () => projections._2))
      }
      val assertion = tf.assert(
        tf.any(tf.stack(predicates.map(_._1))),
        Seq("No projections found for the provided language pair."))
      val default = () => tf.createWith(controlDependencies = Set(assertion)) {
        tf.identity(projectionsForSize.head._1)
      }
      tf.cases(predicates, default)
    } else {
      ???
    }
  }
}

object WordEmbeddingsPerLanguagePair {
  def apply(
      embeddingsSize: Int,
      mergedEmbeddings: Boolean = false,
      mergedProjections: Boolean = false
  ): WordEmbeddingsPerLanguagePair = {
    new WordEmbeddingsPerLanguagePair(embeddingsSize, mergedEmbeddings, mergedProjections)
  }

  private[WordEmbeddingsPerLanguagePair] def languagePairIdsToId(
      numLanguages: Int,
      srcLanguageId: Output,
      tgtLanguageId: Output
  ): Output = {
    srcLanguageId * (numLanguages - 1) + tgtLanguageId -
        tf.less(srcLanguageId, tgtLanguageId).cast(INT32)
  }
}
