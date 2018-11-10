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

package org.platanios.symphony.mt.experiments.config

import org.platanios.symphony.mt.data.{DataConfig, GeneratedVocabulary}
import org.platanios.symphony.mt.models.parameters._
import org.platanios.tensorflow.api.tf

import com.typesafe.config.Config

/**
  * @author Emmanouil Antonios Platanios
  */
class ParametersConfigParser(
    val dataConfig: DataConfig
) extends ConfigParser[ParametersConfigParser.Parameters] {
  override def parse(config: Config): ParametersConfigParser.Parameters = {
    val wordEmbeddingsSize = config.get[Int]("word-embeddings-size")
    val manager = config.get[String]("manager")
    val sharedWordEmbeddings: Boolean = dataConfig.vocabulary match {
      case GeneratedVocabulary(_, true) => true
      case _ => false
    }
    val wordEmbeddingsType = {
      if (sharedWordEmbeddings)
        SharedWordEmbeddings(wordEmbeddingsSize)
      else if (manager == "pairwise")
        WordEmbeddingsPerLanguagePair(wordEmbeddingsSize)
      else
        WordEmbeddingsPerLanguage(wordEmbeddingsSize)
    }
    val parameterManager = manager match {
      case "pairwise" =>
        PairwiseManager(
          wordEmbeddingsType = wordEmbeddingsType,
          variableInitializer = tf.VarianceScalingInitializer(
            1.0f,
            tf.VarianceScalingInitializer.FanAverageScalingMode,
            tf.VarianceScalingInitializer.UniformDistribution))
      case "contextual-language" =>
        val languageEmbeddingsSize = config.get[Int]("language-embeddings-size")
        val hiddenLayers = config.get[String]("contextual-hidden-layers", default = "")
        LanguageEmbeddingsManager(
          languageEmbeddingsSize = languageEmbeddingsSize,
          wordEmbeddingsType = wordEmbeddingsType,
          hiddenLayers = if (hiddenLayers.nonEmpty) hiddenLayers.split('-').map(_.toInt) else Seq.empty)
      case "contextual-language-pair" =>
        val languageEmbeddingsSize = config.get[Int]("language-embeddings-size")
        val hiddenLayers = config.get[String]("contextual-hidden-layers", default = "")
        LanguageEmbeddingsPairManager(
          languageEmbeddingsSize = languageEmbeddingsSize,
          wordEmbeddingsType = wordEmbeddingsType,
          hiddenLayers = if (hiddenLayers.nonEmpty) hiddenLayers.split('-').map(_.toInt) else Seq.empty)
      case "google-multilingual" =>
        GoogleMultilingualManager(
          wordEmbeddingsType = wordEmbeddingsType,
          variableInitializer = tf.VarianceScalingInitializer(
            1.0f,
            tf.VarianceScalingInitializer.FanAverageScalingMode,
            tf.VarianceScalingInitializer.UniformDistribution))
      case _ => throw new IllegalArgumentException(s"'$manager' does not represent a valid parameter manager type.")
    }
    ParametersConfigParser.Parameters(wordEmbeddingsSize, parameterManager)
  }

  override def tag(config: Config, parsedValue: => ParametersConfigParser.Parameters): Option[String] = {
    val wordEmbeddingsSize = config.get[Int]("word-embeddings-size")
    val manager = config.get[String]("manager")
    val parameterManager = manager match {
      case "pairwise" => "pairwise"
      case "contextual-language" =>
        val languageEmbeddingsSize = config.get[Int]("language-embeddings-size")
        val hiddenLayers = config.get[String]("contextual-hidden-layers", default = "")
        if (hiddenLayers.isEmpty)
          s"contextual-language:$languageEmbeddingsSize"
        else
          s"contextual-language:$languageEmbeddingsSize:$hiddenLayers"
      case "contextual-language-pair" =>
        val languageEmbeddingsSize = config.get[Int]("language-embeddings-size")
        val hiddenLayers = config.get[String]("contextual-hidden-layers", default = "")
        if (hiddenLayers.isEmpty)
          s"contextual-language-pair:$languageEmbeddingsSize"
        else
          s"contextual-language-pair:$languageEmbeddingsSize:$hiddenLayers"
      case "google-multilingual" => "google-multilingual"
      case _ => throw new IllegalArgumentException(s"'$manager' does not represent a valid parameter manager type.")
    }
    Some(s"w:$wordEmbeddingsSize.pm:$parameterManager")
  }
}

object ParametersConfigParser {
  case class Parameters(wordEmbeddingsSize: Int, parameterManager: ParameterManager)
}
