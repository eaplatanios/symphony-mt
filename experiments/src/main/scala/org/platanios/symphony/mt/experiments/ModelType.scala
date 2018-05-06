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

  def getParametersManager(
      languageEmbeddingsSize: Int,
      wordEmbeddingsSize: Int,
      sharedWordEmbeddings: Boolean
  ): ParameterManager

  override def toString: String
}

object ModelType {
  implicit val modelTypeRead: scopt.Read[ModelType] = {
    scopt.Read.reads(value => {
      value.split(":") match {
        case Array("pairwise") => Pairwise
        case Array("hyper_lang") => HyperLanguage()
        case Array("hyper_lang", hiddenLayers) => HyperLanguage(hiddenLayers.split('-').map(_.toInt))
        case Array("hyper_lang_pair") => HyperLanguagePair()
        case Array("hyper_lang_pair", hiddenLayers) => HyperLanguagePair(hiddenLayers.split('-').map(_.toInt))
        case Array("google_multilingual") => GoogleMultilingual
        case _ => throw new IllegalArgumentException(s"'$value' does not represent a valid model type.")
      }
    })
  }
}

case object Pairwise extends ModelType {
  override val name: String = "pairwise"

  override def getParametersManager(
      languageEmbeddingsSize: Int,
      wordEmbeddingsSize: Int,
      sharedWordEmbeddings: Boolean
  ): ParameterManager = {
    val wordEmbeddingsType = {
      if (sharedWordEmbeddings)
        SharedWordEmbeddings(wordEmbeddingsSize)
      else
        WordEmbeddingsPerLanguagePair(wordEmbeddingsSize)
    }
    PairwiseManager(
      wordEmbeddingsType = wordEmbeddingsType,
      variableInitializer = tf.VarianceScalingInitializer(
        1.0f,
        tf.VarianceScalingInitializer.FanAverageScalingMode,
        tf.VarianceScalingInitializer.UniformDistribution))
  }

  override def toString: String = "pairwise"
}

case class HyperLanguage(hiddenLayers: Seq[Int] = Seq.empty) extends ModelType {
  override val name: String = "hyper_lang"

  override def getParametersManager(
      languageEmbeddingsSize: Int,
      wordEmbeddingsSize: Int,
      sharedWordEmbeddings: Boolean
  ): ParameterManager = {
    val wordEmbeddingsType = {
      if (sharedWordEmbeddings)
        SharedWordEmbeddings(wordEmbeddingsSize)
      else
        WordEmbeddingsPerLanguage(wordEmbeddingsSize)
    }
    LanguageEmbeddingsManager(
      languageEmbeddingsSize = languageEmbeddingsSize,
      wordEmbeddingsType = wordEmbeddingsType,
      hiddenLayers = hiddenLayers)
  }

  override def toString: String = "hyper_lang"
}

case class HyperLanguagePair(hiddenLayers: Seq[Int] = Seq.empty) extends ModelType {
  override val name: String = "hyper_lang_pair"

  override def getParametersManager(
      languageEmbeddingsSize: Int,
      wordEmbeddingsSize: Int,
      sharedWordEmbeddings: Boolean
  ): ParameterManager = {
    val wordEmbeddingsType = {
      if (sharedWordEmbeddings)
        SharedWordEmbeddings(wordEmbeddingsSize)
      else
        WordEmbeddingsPerLanguage(wordEmbeddingsSize)
    }
    LanguageEmbeddingsPairManager(
      languageEmbeddingsSize = languageEmbeddingsSize,
      wordEmbeddingsType = wordEmbeddingsType,
      hiddenLayers = hiddenLayers)
  }

  override def toString: String = "hyper_lang_pair"
}

case object GoogleMultilingual extends ModelType {
  override val name: String = "google_multilingual"

  override def getParametersManager(
      languageEmbeddingsSize: Int,
      wordEmbeddingsSize: Int,
      sharedWordEmbeddings: Boolean
  ): ParameterManager = {
    val wordEmbeddingsType = {
      if (sharedWordEmbeddings)
        SharedWordEmbeddings(wordEmbeddingsSize)
      else
        WordEmbeddingsPerLanguage(wordEmbeddingsSize)
    }
    GoogleMultilingualManager(
      wordEmbeddingsType = wordEmbeddingsType,
      variableInitializer = tf.VarianceScalingInitializer(
        1.0f,
        tf.VarianceScalingInitializer.FanAverageScalingMode,
        tf.VarianceScalingInitializer.UniformDistribution))
  }

  override def toString: String = "google_multilingual"
}
