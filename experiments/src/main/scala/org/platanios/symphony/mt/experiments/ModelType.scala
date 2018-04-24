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

package org.platanios.symphony.mt.experiments

import org.platanios.symphony.mt.models.parameters._
import org.platanios.tensorflow.api._

/**
  * @author Emmanouil Antonios Platanios
  */
sealed trait ModelType {
  val name: String

  def getParametersManager(languageEmbeddingsSize: Int, wordEmbeddingsSize: Int): ParameterManager

  override def toString: String
}

object ModelType {
  implicit val modelTypeRead: scopt.Read[ModelType] = {
    scopt.Read.reads {
      case "pairwise" => Pairwise
      case "hyper_lang" => HyperLanguage
      case "hyper_lang_pair" => HyperLanguagePair
      case "google_multilingual" => GoogleMultilingual
      case value => throw new IllegalArgumentException(s"'$value' does not represent a valid model type.")
    }
  }
}

case object Pairwise extends ModelType {
  override val name: String = "pairwise"

  override def getParametersManager(languageEmbeddingsSize: Int, wordEmbeddingsSize: Int): ParameterManager = {
    ParameterManager(
      wordEmbeddingsSize = wordEmbeddingsSize,
      variableInitializer = tf.VarianceScalingInitializer(
        1.0f,
        tf.VarianceScalingInitializer.FanAverageScalingMode,
        tf.VarianceScalingInitializer.UniformDistribution))
  }

  override def toString: String = "pairwise"
}

case object HyperLanguage extends ModelType {
  override val name: String = "hyper_lang"

  override def getParametersManager(languageEmbeddingsSize: Int, wordEmbeddingsSize: Int): ParameterManager = {
    LanguageEmbeddingsManager(
      languageEmbeddingsSize = languageEmbeddingsSize,
      wordEmbeddingsSize = wordEmbeddingsSize)
  }

  override def toString: String = "hyper_lang"
}

case object HyperLanguagePair extends ModelType {
  override val name: String = "hyper_lang_pair"

  override def getParametersManager(languageEmbeddingsSize: Int, wordEmbeddingsSize: Int): ParameterManager = {
    LanguageEmbeddingsPairManager(
      languageEmbeddingsSize = languageEmbeddingsSize,
      wordEmbeddingsSize = wordEmbeddingsSize)
  }

  override def toString: String = "hyper_lang_pair"
}

case object GoogleMultilingual extends ModelType {
  override val name: String = "google_multilingual"

  override def getParametersManager(languageEmbeddingsSize: Int, wordEmbeddingsSize: Int): ParameterManager = {
    GoogleMultilingualManager(
      wordEmbeddingsSize = wordEmbeddingsSize,
      variableInitializer = tf.VarianceScalingInitializer(
        1.0f,
        tf.VarianceScalingInitializer.FanAverageScalingMode,
        tf.VarianceScalingInitializer.UniformDistribution))
  }

  override def toString: String = "google_multilingual"
}
