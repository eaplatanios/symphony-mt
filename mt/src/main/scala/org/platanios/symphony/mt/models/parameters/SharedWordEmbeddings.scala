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
class SharedWordEmbeddings protected (
    override val embeddingsSize: Int
) extends WordEmbeddingsType {
  override type T = Output

  override def createStringToIndexLookupTable(languages: Seq[(Language, Vocabulary)]): Output = {
    languages.head._2.stringToIndexLookupTable(name = "SharedStringToIndexLookupTable").handle
  }

  override def createIndexToStringLookupTable(languages: Seq[(Language, Vocabulary)]): Output = {
    languages.head._2.indexToStringLookupTable(name = "SharedIndexToStringLookupTable").handle
  }

  override def createWordEmbeddings(languages: Seq[(Language, Vocabulary)]): Output = {
    val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
    val someLanguage = languages.head
    tf.variable(
      someLanguage._1.name, FLOAT32, Shape(someLanguage._2.size, embeddingsSize),
      embeddingsInitializer).value
  }

  override def lookupTable(lookupTable: Output, languageId: Output): Output = {
    lookupTable
  }

  override def embeddingLookup(
      embeddingTables: Output,
      languageIds: Seq[Output],
      languageId: Output,
      keys: Output
  )(implicit context: Output): Output = {
    embeddingTables.gather(keys)
  }

  override def projectionToWords(
      languages: Seq[(Language, Vocabulary)],
      languageIds: Seq[Output],
      projectionsToWords: mutable.Map[Int, Output],
      inputSize: Int,
      languageId: Output
  )(implicit context: Output): Output = {
    projectionsToWords
        .getOrElseUpdate(inputSize, {
          val weightsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
          val someLanguage = languages.head
          tf.variable(
            s"${someLanguage._1.name}/OutWeights", FLOAT32,
            Shape(inputSize, someLanguage._2.size), weightsInitializer).value
        })
  }
}

object SharedWordEmbeddings {
  def apply(embeddingsSize: Int): SharedWordEmbeddings = {
    new SharedWordEmbeddings(embeddingsSize)
  }
}
