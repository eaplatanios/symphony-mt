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
import org.platanios.tensorflow.api.core.exception.InvalidDataTypeException
import org.platanios.tensorflow.api.ops.{FunctionGraph, Op}

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
class ParameterManager protected (
    val wordEmbeddingsSize: Int,
    val mergedWordEmbeddings: Boolean = false,
    val mergedWordProjections: Boolean = false,
    val variableInitializer: tf.VariableInitializer = null
) {
  protected var environment  : Environment                 = _
  protected var deviceManager: Option[DeviceManager]       = None
  protected var languages    : Seq[(Language, Vocabulary)] = _

  protected val languageIds                : mutable.Map[Graph, Seq[Output]] = mutable.Map.empty
  protected val stringToIndexLookupTables  : mutable.Map[Graph, Output]      = mutable.Map.empty
  protected val stringToIndexLookupDefaults: mutable.Map[Graph, Output]      = mutable.Map.empty
  protected val indexToStringLookupTables  : mutable.Map[Graph, Output]      = mutable.Map.empty
  protected val indexToStringLookupDefaults: mutable.Map[Graph, Output]      = mutable.Map.empty
  protected val wordEmbeddings             : mutable.Map[Graph, Seq[Output]] = mutable.Map.empty

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
    stringToIndexLookupDefaults -= graph
    indexToStringLookupTables -= graph
    indexToStringLookupDefaults -= graph
    wordEmbeddings -= graph
    projectionsToWords -= graph
  }

  def initialize(languages: Seq[(Language, Vocabulary)]): Unit = {
    tf.createWithVariableScope("ParameterManager/") {
      languageIds.keys.filter(_.isClosed).foreach(removeGraph)
      this.languages = languages
      val graph = currentGraph
      if (!languageIds.contains(graph)) {
        languageIds += graph -> tf.createWithVariableScope("LanguageIDs/") {
          languages.map(_._1).zipWithIndex.map(l => tf.constant(l._2, name = l._1.name))
        }

        tf.createWithVariableScope("StringToIndexLookupTables/") {
          val tables = languages.map(l => l._2.stringToIndexLookupTable(name = l._1.name))
          stringToIndexLookupTables += graph -> tf.stack(tables.map(_.handle))
          stringToIndexLookupDefaults += graph -> tf.constant(Vocabulary.UNKNOWN_TOKEN_ID, INT64, name = "Default")
          tables
        }

        tf.createWithVariableScope("IndexToStringLookupTables/") {
          val tables = languages.map(l => l._2.indexToStringLookupTable(name = l._1.name))
          indexToStringLookupTables += graph -> tf.stack(tables.map(_.handle))
          indexToStringLookupDefaults += graph -> tf.constant(Vocabulary.UNKNOWN_TOKEN, STRING, name = "Default")
        }

        wordEmbeddings += graph -> tf.createWithVariableScope("WordEmbeddings/") {
          val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
          if (!mergedWordEmbeddings) {
            languages.map(l =>
              tf.variable(l._1.name, FLOAT32, Shape(l._2.size, wordEmbeddingsSize), embeddingsInitializer).value)
          } else {
            val vocabSizes = languages.map(_._2.size)
            val merged = tf.variable(
              "Embeddings", FLOAT32, Shape(vocabSizes.sum, wordEmbeddingsSize), embeddingsInitializer).value
            val sizes = tf.createWithNameScope("VocabularySizes")(tf.stack(vocabSizes.map(tf.constant(_))))
            val offsets = tf.concatenate(Seq(tf.zeros(sizes.dataType, Shape(1)), tf.cumsum(sizes)(0 :: -1)))
            Seq(merged, offsets)
          }
        }
      }
    }
  }

  def stringToIndexLookup(languageId: Output): (Output) => Output = (keys: Output) => {
    tf.createWithVariableScope("ParameterManager/StringToIndexLookupTables/") {
      val graph = currentGraph
      ParameterManager.lookup(
        handle = stringToIndexLookupTables(graph).gather(languageId),
        keys = keys,
        defaultValue = stringToIndexLookupDefaults(graph))
    }
  }

  def indexToStringLookup(languageId: Output): (Output) => Output = (keys: Output) => {
    tf.createWithVariableScope("ParameterManager/IndexToStringLookupTables/") {
      val graph = currentGraph
      ParameterManager.lookup(
        handle = indexToStringLookupTables(graph).gather(languageId),
        keys = keys,
        defaultValue = indexToStringLookupDefaults(graph))
    }
  }

  def wordEmbeddings(languageId: Output): (Output) => Output = (keys: Output) => {
    tf.createWithVariableScope("ParameterManager/WordEmbeddings/") {
      val graph = currentGraph
      if (!mergedWordEmbeddings) {
        val predicates = wordEmbeddings(graph).zip(languageIds(graph)).map {
          case (embeddings, langId) => (tf.equal(languageId, langId), () => embeddings)
        }
        val default = () => wordEmbeddings(graph).head
        tf.cases(predicates, default).gather(keys)
      } else {
        val merged = wordEmbeddings(graph)(0)
        val offsets = wordEmbeddings(graph)(1)
        merged.gather(keys + offsets.gather(languageId))
      }
    }
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
    tf.createWithVariableScope("ParameterManager/") {
      tf.variable(name, dataType, shape, initializer = variableInitializer, reuse = variableReuse).value
    }
  }

  def getProjectionToWords(inputSize: Int, languageId: Output): Output = {
    tf.createWithVariableScope("ParameterManager/ProjectionToWords/") {
      val graph = currentGraph
      if (!mergedWordProjections) {
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
      } else {
        val projectionsForSize = projectionsToWords
            .getOrElseUpdate(graph, mutable.HashMap.empty)
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
}

object ParameterManager {
  def apply(
      wordEmbeddingsSize: Int,
      mergedWordEmbeddings: Boolean = false,
      mergedWordProjections: Boolean = false,
      variableInitializer: tf.VariableInitializer = null
  ): ParameterManager = {
    new ParameterManager(wordEmbeddingsSize, mergedWordEmbeddings, mergedWordProjections, variableInitializer)
  }

  /** Creates an op that looks up the provided keys in the lookup table referred to by `handle` and returns the
    * corresponding values.
    *
    * @param  handle `RESOURCE` tensor containing a handle to the lookup table.
    * @param  keys   Tensor containing the keys to look up.
    * @param  name   Name for the created op.
    * @return Created op output.
    * @throws InvalidDataTypeException If the provided keys data types does not match the keys data type of this table.
    */
  @throws[InvalidDataTypeException]
  private[ParameterManager] def lookup(
      handle: Output,
      keys: Output,
      defaultValue: Output,
      name: String = "Lookup"
  ): Output = tf.createWithNameScope(name) {
    val values = Op.Builder("LookupTableFindV2", name)
        .addInput(handle)
        .addInput(keys)
        .addInput(defaultValue)
        .build().outputs(0)
    values.setShape(keys.shape)
    values
  }
}

class LanguageEmbeddingsPairParameterManager protected (
    val languageEmbeddingsSize: Int,
    override val wordEmbeddingsSize: Int,
    override val mergedWordEmbeddings: Boolean = false,
    override val mergedWordProjections: Boolean = false,
    override val variableInitializer: tf.VariableInitializer = null
) extends ParameterManager(wordEmbeddingsSize, mergedWordEmbeddings, mergedWordProjections, variableInitializer) {
  protected val languageEmbeddings: mutable.Map[Graph, Output]                      = mutable.Map.empty
  protected val parameters        : mutable.Map[Graph, mutable.Map[String, Output]] = mutable.Map.empty

  override protected def removeGraph(graph: Graph): Unit = {
    super.removeGraph(graph)
    languageEmbeddings -= graph
    parameters -= graph
  }

  override def initialize(languages: Seq[(Language, Vocabulary)]): Unit = {
    tf.createWithVariableScope("ParameterManager/") {
      super.initialize(languages)
      val graph = currentGraph
      if (!languageEmbeddings.contains(graph)) {
        languageEmbeddings += graph -> tf.createWithVariableScope("LanguageEmbeddings/") {
          val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
          tf.variable(
            "LanguageEmbeddings", FLOAT32, Shape(languages.length, languageEmbeddingsSize),
            initializer = embeddingsInitializer).value
        }
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
    tf.createWithVariableScope("ParameterManager/") {
      val graph = currentGraph
      val variableScopeName = tf.currentVariableScope.name
      val fullName = if (variableScopeName != null && variableScopeName != "") s"$variableScopeName/$name" else name

      def create(): Output = tf.createWithVariableScope(name) {
        val languagePair = tf.stack(Seq(context.get._1, context.get._2))
        val embeddings = languageEmbeddings(graph).gather(languagePair).reshape(Shape(1, -1))
        val weights = tf.variable("Dense/Weights", FLOAT32, Shape(2 * languageEmbeddingsSize, shape.numElements.toInt))
        val bias = tf.variable("Dense/Bias", FLOAT32, Shape(shape.numElements.toInt))
        val parameters = tf.linear(embeddings, weights, bias, "Dense")
        parameters.cast(dataType).reshape(shape)
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
}

object LanguageEmbeddingsPairParameterManager {
  def apply(
      languageEmbeddingsSize: Int,
      wordEmbeddingsSize: Int,
      mergedWordEmbeddings: Boolean = false,
      mergedWordProjections: Boolean = false,
      variableInitializer: tf.VariableInitializer = null
  ): LanguageEmbeddingsPairParameterManager = {
    new LanguageEmbeddingsPairParameterManager(
      languageEmbeddingsSize,
      wordEmbeddingsSize,
      mergedWordEmbeddings,
      mergedWordProjections,
      variableInitializer)
  }
}

class LanguageEmbeddingsParameterManager protected (
    val languageEmbeddingsSize: Int,
    override val wordEmbeddingsSize: Int,
    override val mergedWordEmbeddings: Boolean = false,
    override val mergedWordProjections: Boolean = false,
    override val variableInitializer: tf.VariableInitializer = null
) extends ParameterManager(wordEmbeddingsSize, mergedWordEmbeddings, mergedWordProjections, variableInitializer) {
  protected val languageEmbeddings: mutable.Map[Graph, Output]                      = mutable.Map.empty
  protected val parameters        : mutable.Map[Graph, mutable.Map[String, Output]] = mutable.Map.empty

  override protected def removeGraph(graph: Graph): Unit = {
    super.removeGraph(graph)
    languageEmbeddings -= graph
    parameters -= graph
  }

  override def initialize(languages: Seq[(Language, Vocabulary)]): Unit = {
    tf.createWithVariableScope("ParameterManager/") {
      super.initialize(languages)
      val graph = currentGraph
      if (!languageEmbeddings.contains(graph)) {
        languageEmbeddings += graph -> tf.createWithVariableScope("LanguageEmbeddings/") {
          val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
          tf.variable(
            "LanguageEmbeddings", FLOAT32, Shape(languages.length, languageEmbeddingsSize),
            initializer = embeddingsInitializer).value
        }
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
    tf.createWithVariableScope("ParameterManager/") {
      val graph = currentGraph
      val variableScopeName = tf.currentVariableScope.name
      val fullName = if (variableScopeName != null && variableScopeName != "") s"$variableScopeName/$name" else name

      def create(): Output = tf.createWithVariableScope(name) {
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
}

object LanguageEmbeddingsParameterManager {
  def apply(
      languageEmbeddingsSize: Int,
      wordEmbeddingsSize: Int,
      mergedWordEmbeddings: Boolean = false,
      mergedWordProjections: Boolean = false,
      variableInitializer: tf.VariableInitializer = null
  ): LanguageEmbeddingsParameterManager = {
    new LanguageEmbeddingsParameterManager(
      languageEmbeddingsSize,
      wordEmbeddingsSize,
      mergedWordEmbeddings,
      mergedWordProjections,
      variableInitializer)
  }
}
