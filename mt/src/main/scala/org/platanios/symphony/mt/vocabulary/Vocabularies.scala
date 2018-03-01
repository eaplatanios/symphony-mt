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

package org.platanios.symphony.mt.vocabulary

import org.platanios.symphony.mt.Language
import org.platanios.tensorflow.api._

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
class Vocabularies protected (
    val languages: Map[Language, Vocabulary],
    val embeddingsSize: Int
) {
  lazy val vocabularies: Seq[Vocabulary] = languages.values.toSeq

  protected lazy val languageIds: Seq[Output] = tf.createWithNameScope("Vocabularies/LanguageIDs") {
    languages.keys.zipWithIndex.map(l => tf.constant(l._2, name = s"${l._1}ID")).toSeq
  }

  protected lazy val lookupTables: Seq[tf.HashTable] = tf.createWithNameScope("Vocabularies/LookupTables") {
    vocabularies.map(_.lookupTable())
  }

  protected lazy val wordEmbeddings: Seq[Output] = tf.createWithNameScope("Vocabularies/WordEmbeddings") {
    val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
    languages.map(l =>
      tf.variable(s"${l._1.name}", FLOAT32, Shape(l._2.size, embeddingsSize), embeddingsInitializer).value).toSeq
  }

  protected val projections: mutable.Map[Int, Seq[Output]] = mutable.Map.empty[Int, Seq[Output]]

  def lookupTable(languageId: Output): (Output) => Output = (keys: Output) => {
    val predicates = lookupTables.zip(languageIds).map {
      case (table, langId) => (tf.equal(languageId, langId), () => table.lookup(keys))
    }
    val default = () => lookupTables.head.lookup(keys)
    tf.cases(predicates, default)
  }

  def embeddings(languageId: Output): Output = {
    val predicates = wordEmbeddings.zip(languageIds).map {
      case (embeddings, langId) => (tf.equal(languageId, langId), () => embeddings)
    }
    val default = () => wordEmbeddings.head
    tf.cases(predicates, default)
  }

  def projection(inputSize: Int, languageId: Output): Output = {
    val projectionsForSize = projections.getOrElseUpdate(inputSize, {
      val weightsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
      languages.map(l =>
        tf.variable(s"${l._1.name}/OutWeights", FLOAT32, Shape(inputSize, l._2.size), weightsInitializer).value).toSeq
    })
    val predicates = projectionsForSize.zip(languageIds).map {
      case (projs, langId) => (tf.equal(languageId, langId), () => projs)
    }
    val default = () => projectionsForSize.head
    tf.cases(predicates, default)
  }
}

object Vocabularies {
  def apply(languages: Map[Language, Vocabulary], embeddingsSize: Int): Vocabularies = {
    new Vocabularies(languages, embeddingsSize)
  }

  case class Table private[Vocabularies] (handle: Output, defaultValue: Output) {
    /** Creates an op that looks up the provided keys in this table and returns the corresponding values.
      *
      * @param  keys Tensor containing the keys to look up.
      * @param  name Name for the created op.
      * @return Created op output.
      */
    def lookup(keys: Output, name: String = "Lookup"): Output = tf.createWithNameScope(name) {
      val values = tf.Op.Builder("LookupTableFindV2", name)
          .addInput(handle)
          .addInput(keys)
          .addInput(defaultValue)
          .build().outputs(0)
      values.setShape(keys.shape)
      values
    }
  }
}
