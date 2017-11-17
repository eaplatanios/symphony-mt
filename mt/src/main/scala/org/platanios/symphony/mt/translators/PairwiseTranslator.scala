/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.symphony.mt.translators

import org.platanios.symphony.mt.core.{Configuration, Language, Translator}
import org.platanios.symphony.mt.data.Datasets.{MTLayer, MTTextLinesDataset, MTTrainLayer}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.StopCriteria

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class PairwiseTranslator(override protected var configuration: Configuration = Configuration())
    extends Translator(configuration) {
  // Create the input and the train input parts of the model.
  private[this] val seqShape   = Shape(configuration.batchSize, -1)
  private[this] val lenShape   = Shape(configuration.batchSize)
  private[this] val input      = tf.learn.Input((INT32, INT32), (seqShape, lenShape))
  private[this] val trainInput = tf.learn.Input((INT32, INT32, INT32), (seqShape, seqShape, lenShape))

  override def train(datasets: Seq[Translator.DatasetPair], stopCriteria: StopCriteria): Unit = {
    type LanguagePair = (Language, Language)


    ???
  }

  override def translate(
      sourceLanguage: Language,
      targetLanguage: Language,
      dataset: MTTextLinesDataset
  ): MTTextLinesDataset = ???

  protected def translationTrainLayer(
      sourceVocabularySize: Int,
      targetVocabularySize: Int,
      sourceVocabularyTable: tf.LookupTable,
      targetVocabularyTable: tf.LookupTable
  ): MTTrainLayer

  protected def translationLayer(
      sourceVocabularySize: Int,
      targetVocabularySize: Int,
      sourceVocabularyTable: tf.LookupTable,
      targetVocabularyTable: tf.LookupTable
  ): MTLayer
}
