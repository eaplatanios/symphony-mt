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

/**
  * @author Emmanouil Antonios Platanios
  */
class WordEmbeddingsPerLanguage protected (
    override val embeddingsSize: Int,
    val mergedProjections: Boolean
) extends WordEmbeddingsType {
  override type T = tf.TensorArray

  override def createStringToIndexLookupTable(languages: Seq[(Language, Vocabulary)]): Output = {
    val tables = languages.map(l => l._2.stringToIndexLookupTable(name = l._1.name))
    tf.stack(tables.map(_.handle))
  }

  override def createIndexToStringLookupTable(languages: Seq[(Language, Vocabulary)]): Output = {
    val tables = languages.map(l => l._2.indexToStringLookupTable(name = l._1.name))
    tf.stack(tables.map(_.handle))
  }

  override def createWordEmbeddings(languages: Seq[(Language, Vocabulary)]): tf.TensorArray = {
    val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
    val tensorArray = tf.TensorArray.create(
      size = languages.size,
      dataType = FLOAT32,
      dynamicSize = false,
      clearAfterRead = false,
      inferShape = false)
    languages.zipWithIndex.foldLeft(tensorArray) {
      case (ta, (language, index)) =>
        ta.write(index, tf.variable(
          language._1.name, FLOAT32, Shape(language._2.size, embeddingsSize), embeddingsInitializer).value)
    }
  }

  override def lookupTable(lookupTable: Output, languageId: Output): Output = {
    lookupTable.gather(languageId)
  }

  override def embeddingLookup(
      embeddingTables: tf.TensorArray,
      languageIds: Seq[Output],
      languageId: Output,
      keys: Output,
      context: Option[(Output, Output)]
  ): Output = {
    embeddingTables.read(languageId).gather(keys)
  }

  override def projectionToWords(
      languages: Seq[(Language, Vocabulary)],
      languageIds: Seq[Output],
      projectionsToWords: mutable.Map[Int, Seq[Output]],
      inputSize: Int,
      languageId: Output
  ): Output = {
    if (!mergedProjections) {
      val projectionsForSize = projectionsToWords
          .getOrElseUpdate(inputSize, {
            val weightsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
            languages.map(l => tf.variable(
              s"${l._1.name}/OutWeights", FLOAT32, Shape(inputSize, l._2.size), weightsInitializer).value)
          })
      val predicates = projectionsForSize.zip(languageIds).map {
        case (projections, langId) => (tf.equal(languageId, langId), () => projections)
      }
      val default = () => projectionsForSize.head
      tf.cases(predicates, default)
    } else {
      val projectionsForSize = projectionsToWords
          .getOrElseUpdate(inputSize, {
            val weightsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
            val vocabSizes = languages.map(_._2.size)
            val merged = tf.variable(
              "ProjectionWeights", FLOAT32, Shape(inputSize, vocabSizes.sum), weightsInitializer).value
            val sizes = tf.createWithNameScope("VocabularySizes")(tf.stack(vocabSizes.map(tf.constant(_))))
            val offsets = tf.concatenate(Seq(tf.zeros(sizes.dataType, Shape(1)), tf.cumsum(sizes)(0 :: -1)))
            Seq(merged, offsets, sizes)
          })
      val merged = projectionsForSize(0)
      val offsets = projectionsForSize(1)
      val sizes = projectionsForSize(2)
      tf.slice(
        merged,
        tf.stack(Seq(0, offsets.gather(languageId))),
        tf.stack(Seq(inputSize, sizes.gather(languageId))))
    }
  }
}

object WordEmbeddingsPerLanguage {
  def apply(
      embeddingsSize: Int,
      mergedProjections: Boolean = false
  ): WordEmbeddingsPerLanguage = {
    new WordEmbeddingsPerLanguage(embeddingsSize, mergedProjections)
  }
}
