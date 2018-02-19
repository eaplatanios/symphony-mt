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

import org.platanios.symphony.mt.{Environment, Language, LogConfig}
import org.platanios.symphony.mt.data.{DataConfig, ParallelDataset}
import org.platanios.symphony.mt.models.attention.{AttentionCommon, FixedSinusoidPositionEmbeddings, PositionEmbeddings}
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.{Mode, StopCriteria}
import org.platanios.tensorflow.api.learn.layers.{Input, Layer}

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class MultilingualModel protected (
    val name: String,
    val languages: Seq[Language],
    val vocabularies: Seq[Vocabulary]
) {
  val dataConfig: DataConfig = DataConfig()
  val logConfig : LogConfig  = LogConfig()

  val dataType             : DataType = FLOAT32
  val languageEmbeddingSize: Int      = 512

  // TODO: Make this configurable.
  val parameterManager: ParametersManager    = DefaultParametersManager
  val useSelfAttentionProximityBias: Boolean = false
  val positionEmbeddings: PositionEmbeddings = FixedSinusoidPositionEmbeddings(1.0f, 1e4f)
  val postPositionEmbeddingsDropout: Float   = 0.1f

  /** Each input consists of a tuple containing:
    *   - The language ID. TODO: !!!
    *   - A tensor containing a padded batch of sentences consisting of word IDs.
    *   - A tensor containing the sentence lengths for the aforementioned padded batch.
    */
  protected val input      = Input((INT32, INT32), (Shape(-1, -1), Shape(-1)))
  protected val trainInput = Input((INT32, INT32), (Shape(-1, -1), Shape(-1)))

  def train(dataset: ParallelDataset, stopCriteria: StopCriteria): Unit

  private final def trainLayer: Layer[((Output, Output), (Output, Output)), (Output, Output)] = {
    new Layer[((Output, Output), (Output, Output)), (Output, Output)](name) {
      override val layerType: String = "TrainLayer"

      override protected def _forward(
          input: ((Output, Output), (Output, Output)),
          mode: Mode
      ): (Output, Output) = tf.createWithVariableScope(name) {
        // val langEmbeddings = parameter("LanguageEmbeddings", dataType, Shape(languages.length, languageEmbeddingSize), forceVariable = true)

        tf.createWithVariableScope("Encoder") {

        }

        tf.createWithVariableScope("Decoder") {
          // val decLang = input._2._1
          // val decLangEmbedding = tf.embeddingLookup(langEmbeddings, decLang)


          // TODO: !!!
        }


        ???
      }
    }
  }

  protected def encoder[S](input: (Output, Output), mode: Mode): S = {
    val padding = tf.sequenceMask(input._2, dataType = FLOAT32, name = "Padding")
    val attentionBias = AttentionCommon.attentionBiasIgnorePadding(padding)
    var encoderSelfAttentionBias = attentionBias
    val encoderDecoderAttentionBias = attentionBias
    if (useSelfAttentionProximityBias)
      encoderSelfAttentionBias += AttentionCommon.attentionBiasProximal(tf.shape(input._1)(1))
    // TODO: Optionally add target space embedding for multi-lingual translation.
    var encoderInput = positionEmbeddings.add(input._1)
    if (mode.isTraining)
      encoderInput = tf.dropout(encoderInput, 1.0f - postPositionEmbeddingsDropout)



    // val encLang = input._1
    // val encLangEmbedding = tf.embeddingLookup(langEmbeddings, encLang)



    // TODO: !!!


    ???
  }
}
