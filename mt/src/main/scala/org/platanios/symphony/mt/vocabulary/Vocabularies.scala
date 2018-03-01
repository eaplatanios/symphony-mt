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
import org.platanios.tensorflow.api.ops.TensorArray

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
class Vocabularies protected (
    val languages: Map[Language, Vocabulary],
    val embeddingsSize: Int
) {
  val vocabularies: Seq[Vocabulary] = languages.values.toSeq

  protected val (lookupTables, lookupTableHandles, defaultValues, wordEmbeddings) = {
    tf.createWithNameScope("Vocabularies") {
      val numLanguages = tf.constant(vocabularies.size)
      val lookupTables = vocabularies.map(_.lookupTable())
      val lookupTableHandles = tf.createWithNameScope("LookupTables/Handles") {
        var ta = TensorArray.create(
          numLanguages, RESOURCE, clearAfterRead = false, inferShape = false, elementShape = Shape())
        languages.keys.zip(lookupTables).zipWithIndex.foreach {
          case ((l, lt), i) => ta = ta.write(i, lt.handle, name = s"${l.name}/Write")
        }
        ta
      }
      val defaultValues = tf.createWithNameScope("LookupTables/DefaultValues") {
        var ta = TensorArray.create(
          numLanguages, INT64, clearAfterRead = false, inferShape = false, elementShape = Shape())
        languages.keys.zip(lookupTables).zipWithIndex.foreach {
          case ((l, lt), i) => ta = ta.write(i, lt.defaultValue, name = s"${l.name}/Write")
        }
        ta
      }
      val embeddings = tf.createWithNameScope("Embeddings") {
        val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
        var ta = TensorArray.create(
          numLanguages, FLOAT32, clearAfterRead = false, inferShape = false, elementShape = Shape(-1, embeddingsSize))
        languages.zipWithIndex.foreach {
          case ((l, v), i) => ta = ta.write(
            i, tf.variable(l.name, FLOAT32, Shape(v.size, embeddingsSize), embeddingsInitializer).value,
            name = s"${l.name}/Write")
        }
        ta
      }
      (lookupTables, lookupTableHandles, defaultValues, embeddings)
    }
  }

  protected val projections: mutable.Map[Int, TensorArray] = mutable.Map.empty[Int, TensorArray]

  def lookupTable(languageId: Output): Vocabularies.Table = {
    val handle = lookupTableHandles.read(languageId)
    val defaultValue = defaultValues.read(languageId)
    Vocabularies.Table(handle, defaultValue)
  }

  def embeddings(languageId: Output): Output = wordEmbeddings.read(languageId, name = "Embeddings/Read")

  def projection(inputSize: Int, languageId: Output): Output = {
    projections.getOrElseUpdate(inputSize, {
      val weightsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
      var ta = TensorArray.create(
        vocabularies.size, FLOAT32, clearAfterRead = false, inferShape = false, elementShape = Shape(inputSize, -1))
      languages.zipWithIndex.foreach {
        case ((l, v), i) => ta = ta.write(
          i, tf.variable(s"${l.name}/OutWeights", FLOAT32, Shape(inputSize, v.size), weightsInitializer),
          name = s"${l.name}/OutWeights/Write")
      }
      ta
    }).read(languageId)
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
