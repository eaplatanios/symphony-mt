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
import org.platanios.symphony.mt.models.{Context, ModelConfig, Sequences}
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
class GoogleMultilingualManager protected (
    override val wordEmbeddingsType: WordEmbeddingsType,
    override val variableInitializer: tf.VariableInitializer = null
) extends ParameterManager(wordEmbeddingsType, variableInitializer) {
  protected val languageEmbeddings: mutable.Map[Graph, Output[Float]]                    = mutable.Map.empty
  protected val parameters        : mutable.Map[Graph, mutable.Map[String, Output[Any]]] = mutable.Map.empty

  override protected def removeGraph(graph: Graph): Unit = {
    super.removeGraph(graph)
    languageEmbeddings -= graph
    parameters -= graph
  }

  override def initialize(
      languages: Seq[(Language, Vocabulary)],
      modelConfig: ModelConfig
  ): Unit = {
    tf.variableScope("ParameterManager") {
      super.initialize(languages, modelConfig)
      val graph = currentGraph
      if (!languageEmbeddings.contains(graph)) {
        languageEmbeddings += graph -> {
          val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
          tf.variable[Float](
            "LanguageEmbeddings", Shape(languages.length, wordEmbeddingsType.embeddingsSize),
            initializer = embeddingsInitializer).value
        }
      }
    }
  }

  override def postprocessEmbeddedSequences(
      sequences: Sequences[Float]
  )(implicit context: Context): Sequences[Float] = {
    val batchSize = tf.shape(sequences.sequences).slice(0).toInt
    val tgtLanguageEmbedding = languageEmbeddings(currentGraph).gather(context.tgtLanguageID).reshape(Shape(1, 1, -1))
    val tgtLanguageEmbeddingTiled = tf.tile(tgtLanguageEmbedding, tf.stack[Int](Seq(batchSize, 1, 1)))
    val processedSrcSentences = tf.concatenate(Seq(tgtLanguageEmbeddingTiled, sequences.sequences), 1)
    val processedSrcSentenceLengths = sequences.lengths + 1
    Sequences(processedSrcSentences, processedSrcSentenceLengths)
  }
}

object GoogleMultilingualManager {
  def apply(
      wordEmbeddingsType: WordEmbeddingsType,
      variableInitializer: tf.VariableInitializer = null
  ): GoogleMultilingualManager = {
    new GoogleMultilingualManager(wordEmbeddingsType, variableInitializer)
  }
}
