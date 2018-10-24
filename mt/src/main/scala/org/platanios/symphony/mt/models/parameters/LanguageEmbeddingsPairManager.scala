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
import org.platanios.symphony.mt.models.Stage
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.TF

import scala.collection.mutable

// TODO: Add support for an optional language embeddings merge layer.

/**
  * @author Emmanouil Antonios Platanios
  */
class LanguageEmbeddingsPairManager protected (
    val languageEmbeddingsSize: Int,
    override val wordEmbeddingsType: WordEmbeddingsType,
    val hiddenLayers: Seq[Int] = Seq.empty,
    override val variableInitializer: tf.VariableInitializer = null
) extends ParameterManager(wordEmbeddingsType, variableInitializer) {
  protected val languageEmbeddings: mutable.Map[Graph, Output[Float]]                    = mutable.Map.empty
  protected val parameters        : mutable.Map[Graph, mutable.Map[String, Output[Any]]] = mutable.Map.empty

  override protected def removeGraph(graph: Graph): Unit = {
    super.removeGraph(graph)
    languageEmbeddings -= graph
    parameters -= graph
  }

  override def initialize(languages: Seq[(Language, Vocabulary)]): Unit = {
    tf.variableScope("ParameterManager") {
      super.initialize(languages)
      val graph = currentGraph
      if (!languageEmbeddings.contains(graph)) {
        languageEmbeddings += graph -> {
          val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
          tf.variable[Float](
            "LanguageEmbeddings", Shape(languages.length, languageEmbeddingsSize),
            initializer = embeddingsInitializer).value
        }
      }
    }
  }

  override def get[P: TF](
      name: String,
      shape: Shape,
      variableInitializer: tf.VariableInitializer = variableInitializer,
      variableReuse: tf.VariableReuse = tf.ReuseOrCreateNewVariable
  )(implicit stage: Stage, context: Output[Int]): Output[P] = {
    tf.variableScope("ParameterManager") {
      val graph = currentGraph
      val variableScopeName = tf.currentVariableScope.name
      val fullName = if (variableScopeName != null && variableScopeName != "") s"$variableScopeName/$name" else name

      def create(): Output[P] = {
        tf.variableScope(name) {
          val languagePair = tf.stack(Seq(context(0), context(1)))
          val embeddings = languageEmbeddings(graph).gather(languagePair).reshape(Shape(1, -1))
          var inputSize = 2 * languageEmbeddingsSize
          var parameters = embeddings
          hiddenLayers.zipWithIndex.foreach(numUnits => {
            val weights = tf.variable[Float](s"Dense${numUnits._2}/Weights", Shape(inputSize, numUnits._1))
            val bias = tf.variable[Float](s"Dense${numUnits._2}/Bias", Shape(numUnits._1))
            inputSize = numUnits._1
            parameters = tf.addBias(tf.matmul(parameters, weights), bias)
          })
          val weights = tf.variable[Float]("Dense/Weights", Shape(inputSize, shape.numElements.toInt))
          val bias = tf.variable[Float]("Dense/Bias", Shape(shape.numElements.toInt))
          parameters = tf.addBias(tf.matmul(parameters, weights), bias)
          parameters.castTo[P].reshape(shape)
        }
      }

      variableReuse match {
        case tf.ReuseExistingVariableOnly =>
          parameters.getOrElseUpdate(graph, mutable.Map.empty)(fullName).asInstanceOf[Output[P]]
        case tf.CreateNewVariableOnly =>
          // TODO: Kind of hacky.
          val created = create()
          parameters.getOrElseUpdate(graph, mutable.Map.empty) += created.op.input.head.name -> created
          created
        case tf.ReuseOrCreateNewVariable =>
          parameters
              .getOrElseUpdate(graph, mutable.Map.empty)
              .getOrElseUpdate(fullName, create())
              .asInstanceOf[Output[P]]
      }
    }
  }
}

object LanguageEmbeddingsPairManager {
  def apply(
      languageEmbeddingsSize: Int,
      wordEmbeddingsType: WordEmbeddingsType,
      hiddenLayers: Seq[Int] = Seq.empty,
      variableInitializer: tf.VariableInitializer = null
  ): LanguageEmbeddingsPairManager = {
    new LanguageEmbeddingsPairManager(
      languageEmbeddingsSize,
      wordEmbeddingsType,
      hiddenLayers,
      variableInitializer)
  }
}
