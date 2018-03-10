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

package org.platanios.symphony.mt.models

import org.platanios.symphony.mt.{Environment, Language}
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.FunctionGraph

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
class ParameterManager protected (
    val wordEmbeddingsSize: Int,
    val variableInitializer: tf.VariableInitializer = null
) {
  protected var environment  : Environment                 = _
  protected var deviceManager: Option[DeviceManager]       = None
  protected var languages    : Seq[(Language, Vocabulary)] = _

  protected val languageIds              : mutable.Map[Graph, Seq[Output]]       = mutable.Map.empty
  protected val stringToIndexLookupTables: mutable.Map[Graph, Seq[tf.HashTable]] = mutable.Map.empty
  protected val indexToStringLookupTables: mutable.Map[Graph, Seq[tf.HashTable]] = mutable.Map.empty
  protected val wordEmbeddings           : mutable.Map[Graph, Seq[Output]]       = mutable.Map.empty

  protected val projectionsToWords: mutable.Map[Graph, mutable.Map[Int, Seq[Output]]] = mutable.Map.empty

  protected var context: Option[(Output, Output)] = None

  def setEnvironment(environment: Environment): Unit = this.environment = environment
  def setDeviceManager(deviceManager: DeviceManager): Unit = this.deviceManager = Some(deviceManager)

  protected def currentGraph: Graph = {
    var graph = tf.currentGraph
    while (graph.isInstanceOf[FunctionGraph])
      graph = graph.asInstanceOf[FunctionGraph].outerGraph
    graph
  }

  protected def removeGraph(graph: Graph): Unit = {
    languageIds -= graph
    stringToIndexLookupTables -= graph
    indexToStringLookupTables -= graph
    wordEmbeddings -= graph
    projectionsToWords -= graph
  }

  def initialize(languages: Seq[(Language, Vocabulary)]): Unit = {
    languageIds.keys.filter(_.isClosed).foreach(removeGraph)
    this.languages = languages
    val graph = currentGraph
    if (!languageIds.contains(graph)) {
      languageIds += graph -> tf.createWithNameScope("ParameterManager/LanguageIDs") {
        languages.map(_._1).zipWithIndex.map(l => tf.constant(l._2, name = l._1.name))
      }
    }
    if (!stringToIndexLookupTables.contains(graph)) {
      stringToIndexLookupTables += graph -> tf.createWithNameScope("ParameterManager/StringToIndexLookupTables") {
        languages.map(l => l._2.stringToIndexLookupTable(name = l._1.name))
      }
    }
    if (!indexToStringLookupTables.contains(graph)) {
      indexToStringLookupTables += graph -> tf.createWithNameScope("ParameterManager/IndexToStringLookupTables") {
        languages.map(l => l._2.indexToStringLookupTable(name = l._1.name))
      }
    }
    if (!wordEmbeddings.contains(graph)) {
      wordEmbeddings += graph -> tf.createWithNameScope("ParameterManager/WordEmbeddings") {
        val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
        languages.map(l =>
          tf.variable(l._1.name, FLOAT32, Shape(l._2.size, wordEmbeddingsSize), embeddingsInitializer).value)
      }
    }
  }

  def stringToIndexLookup(languageId: Output): (Output) => Output = (keys: Output) => {
    val graph = currentGraph
    val predicates = stringToIndexLookupTables(graph).zip(languageIds(graph)).map {
      case (table, langId) => (tf.equal(languageId, langId), () => table.lookup(keys))
    }
    val default = () => stringToIndexLookupTables(graph).head.lookup(keys)
    tf.cases(predicates, default)
  }

  def indexToStringLookup(languageId: Output): (Output) => Output = (keys: Output) => {
    val graph = currentGraph
    val predicates = indexToStringLookupTables(graph).zip(languageIds(graph)).map {
      case (table, langId) => (tf.equal(languageId, langId), () => table.lookup(keys))
    }
    val default = () => indexToStringLookupTables(graph).head.lookup(keys)
    tf.cases(predicates, default)
  }

  def wordEmbeddings(languageId: Output): Output = {
    val graph = currentGraph
    val predicates = wordEmbeddings(graph).zip(languageIds(graph)).map {
      case (embeddings, langId) => (tf.equal(languageId, langId), () => embeddings)
    }
    val default = () => wordEmbeddings(graph).head
    tf.cases(predicates, default)
  }

  def getContext: Option[(Output, Output)] = this.context
  def setContext(context: (Output, Output)): Unit = this.context = Some(context)
  def resetContext(): Unit = this.context = None

  def get(
      name: String,
      dataType: DataType,
      shape: Shape,
      variableInitializer: tf.VariableInitializer = variableInitializer,
      variableReuse: tf.VariableReuse = tf.ReuseOrCreateNewVariable
  )(implicit stage: Stage): Output = {
    tf.variable(name, dataType, shape, initializer = variableInitializer, reuse = variableReuse).value
  }

  def getProjectionToWords(inputSize: Int, languageId: Output): Output = {
    val graph = currentGraph
    val projectionsForSize = projectionsToWords
        .getOrElseUpdate(graph, mutable.HashMap.empty)
        .getOrElseUpdate(inputSize, {
          val weightsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
          languages.map(l =>
            tf.variable(s"${l._1.name}/OutWeights", FLOAT32, Shape(inputSize, l._2.size), weightsInitializer).value)
        })
    val predicates = projectionsForSize.zip(languageIds(graph)).map {
      case (projections, langId) => (tf.equal(languageId, langId), () => projections)
    }
    val default = () => projectionsForSize.head
    tf.cases(predicates, default)
  }
}

object ParameterManager {
  def apply(wordEmbeddingsSize: Int, variableInitializer: tf.VariableInitializer = null): ParameterManager = {
    new ParameterManager(wordEmbeddingsSize, variableInitializer)
  }
}

class LanguageEmbeddingsPairParameterManager protected (
    val languageEmbeddingsSize: Int,
    override val wordEmbeddingsSize: Int,
    override val variableInitializer: tf.VariableInitializer = null
) extends ParameterManager(wordEmbeddingsSize, variableInitializer) {
  protected val languageEmbeddings: mutable.Map[Graph, Output]                      = mutable.Map.empty
  protected val parameters        : mutable.Map[Graph, mutable.Map[String, Output]] = mutable.Map.empty

  override protected def removeGraph(graph: Graph): Unit = {
    super.removeGraph(graph)
    languageEmbeddings -= graph
    parameters -= graph
  }

  override def initialize(languages: Seq[(Language, Vocabulary)]): Unit = {
    super.initialize(languages)
    val graph = currentGraph
    if (!languageEmbeddings.contains(graph)) {
      languageEmbeddings += graph -> tf.createWithNameScope("ParameterManager/LanguageEmbeddings") {
        val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
        tf.variable(
          "LanguageEmbeddings", FLOAT32, Shape(languages.length, languageEmbeddingsSize),
          initializer = embeddingsInitializer).value
      }
    }
  }

  override def get(
      name: String,
      dataType: DataType,
      shape: Shape,
      variableInitializer: tf.VariableInitializer = variableInitializer,
      variableReuse: tf.VariableReuse = tf.ReuseOrCreateNewVariable
  )(implicit stage: Stage): Output = {
    val graph = currentGraph
    val variableScopeName = tf.currentVariableScope.name
    val fullName = if (variableScopeName != null && variableScopeName != "") s"$variableScopeName/$name" else name

    def create(): Output = tf.createWithVariableScope(name) {
      tf.createWith(device = "/device:CPU:0") {
        val languagePair = tf.stack(Seq(context.get._1, context.get._2))
        val embeddings = languageEmbeddings(graph).gather(languagePair).reshape(Shape(1, -1))
        val weights = tf.variable("Dense/Weights", FLOAT32, Shape(2 * languageEmbeddingsSize, shape.numElements.toInt))
        val bias = tf.variable("Dense/Bias", FLOAT32, Shape(shape.numElements.toInt))
        val parameters = tf.linear(embeddings, weights, bias, "Dense")
        parameters.cast(dataType).reshape(shape)
      }
    }

    variableReuse match {
      case tf.ReuseExistingVariableOnly => parameters.getOrElseUpdate(graph, mutable.Map.empty)(fullName)
      case tf.CreateNewVariableOnly =>
        // TODO: Kind of hacky.
        val created = create()
        parameters.getOrElseUpdate(graph, mutable.Map.empty) += created.op.inputs(0).name -> created
        created
      case tf.ReuseOrCreateNewVariable =>
        parameters
            .getOrElseUpdate(graph, mutable.Map.empty)
            .getOrElseUpdate(fullName, create())
    }
  }
}

object LanguageEmbeddingsPairParameterManager {
  def apply(
      languageEmbeddingsSize: Int,
      wordEmbeddingsSize: Int,
      variableInitializer: tf.VariableInitializer = null
  ): LanguageEmbeddingsPairParameterManager = {
    new LanguageEmbeddingsPairParameterManager(
      languageEmbeddingsSize,
      wordEmbeddingsSize,
      variableInitializer)
  }
}

class LanguageEmbeddingsParameterManager protected (
    val languageEmbeddingsSize: Int,
    override val wordEmbeddingsSize: Int,
    override val variableInitializer: tf.VariableInitializer = null
) extends ParameterManager(wordEmbeddingsSize, variableInitializer) {
  protected val languageEmbeddings: mutable.Map[Graph, Output]                      = mutable.Map.empty
  protected val parameters        : mutable.Map[Graph, mutable.Map[String, Output]] = mutable.Map.empty

  override protected def removeGraph(graph: Graph): Unit = {
    super.removeGraph(graph)
    languageEmbeddings -= graph
    parameters -= graph
  }

  override def initialize(languages: Seq[(Language, Vocabulary)]): Unit = {
    super.initialize(languages)
    val graph = currentGraph
    if (!languageEmbeddings.contains(graph)) {
      languageEmbeddings += graph -> tf.createWithNameScope("ParameterManager/LanguageEmbeddings") {
        val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
        tf.variable(
          "LanguageEmbeddings", FLOAT32, Shape(languages.length, languageEmbeddingsSize),
          initializer = embeddingsInitializer).value
      }
    }
  }

  override def get(
      name: String,
      dataType: DataType,
      shape: Shape,
      variableInitializer: tf.VariableInitializer = variableInitializer,
      variableReuse: tf.VariableReuse = tf.ReuseOrCreateNewVariable
  )(implicit stage: Stage): Output = {
    val graph = currentGraph
    val variableScopeName = tf.currentVariableScope.name
    val fullName = if (variableScopeName != null && variableScopeName != "") s"$variableScopeName/$name" else name

    def create(): Output = tf.createWithVariableScope(name) {
      tf.createWith(device = "/device:CPU:0") {
        val language = stage match {
          case Encoding => context.get._1
          case Decoding => context.get._2
        }
        val embedding = languageEmbeddings(graph).gather(language).reshape(Shape(1, -1))
        val weights = tf.variable("Dense/Weights", FLOAT32, Shape(languageEmbeddingsSize, shape.numElements.toInt))
        val bias = tf.variable("Dense/Bias", FLOAT32, Shape(shape.numElements.toInt))
        val parameters = tf.linear(embedding, weights, bias, "Dense")
        parameters.cast(dataType).reshape(shape)
      }
    }

    variableReuse match {
      case tf.ReuseExistingVariableOnly => parameters.getOrElseUpdate(graph, mutable.Map.empty)(fullName)
      case tf.CreateNewVariableOnly =>
        // TODO: Kind of hacky.
        val created = create()
        parameters.getOrElseUpdate(graph, mutable.Map.empty) += created.op.inputs(0).name -> created
        created
      case tf.ReuseOrCreateNewVariable =>
        parameters
            .getOrElseUpdate(graph, mutable.Map.empty)
            .getOrElseUpdate(fullName, create())
    }
  }
}

object LanguageEmbeddingsParameterManager {
  def apply(
      languageEmbeddingsSize: Int,
      wordEmbeddingsSize: Int,
      variableInitializer: tf.VariableInitializer = null
  ): LanguageEmbeddingsParameterManager = {
    new LanguageEmbeddingsParameterManager(
      languageEmbeddingsSize,
      wordEmbeddingsSize,
      variableInitializer)
  }
}
