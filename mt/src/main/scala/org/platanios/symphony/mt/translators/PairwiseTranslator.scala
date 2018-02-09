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

package org.platanios.symphony.mt.translators

import org.platanios.symphony.mt.{Environment, Language}
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.models.Model
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api.Tensor
import org.platanios.tensorflow.api.learn.StopCriteria

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
class PairwiseTranslator protected (
    val env: Environment,
    override val model: (Language, Vocabulary, Language, Vocabulary, Environment) => Model
) extends Translator(model) {
  protected val models: mutable.Map[(Language, Language), Model] = mutable.Map.empty

  override def train(dataset: LoadedDataset, stopCriteria: StopCriteria): Unit = {
    train(dataset, stopCriteria, trainReverse = true)
  }

  def train(dataset: LoadedDataset, stopCriteria: StopCriteria, trainReverse: Boolean): Unit = {
    val languagePairs = {
      if (trainReverse) {
        // We train models for both possible translation directions.
        dataset.languagePairs.flatMap(p => Seq((p._1, p._2), (p._2, p._1)))
      } else {
        dataset.languagePairs
      }
    }
    languagePairs.foreach {
      case (srcLanguage, tgtLanguage) =>
        val currentDatasetFiles = dataset.files(srcLanguage, tgtLanguage)
        val currentModel = models.getOrElseUpdate(
          (srcLanguage, tgtLanguage),
          model(srcLanguage, currentDatasetFiles.srcVocab, tgtLanguage, currentDatasetFiles.tgtVocab, env))
        val currentDataset = () => {
          currentDatasetFiles.createTrainDataset(
            TRAIN_DATASET, repeat = true, dataConfig = currentDatasetFiles.dataConfig)
        }
        currentModel.train(currentDataset, stopCriteria)
    }
  }

  @throws[IllegalStateException]
  override def translate(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataset: () => MTInferDataset
  ): Iterator[((Tensor, Tensor), (Tensor, Tensor))] = {
    models.get((srcLanguage, tgtLanguage)) match {
      case None => // TODO: Maybe instead of throwing an exception we want to let an untrained model make predictions.
        throw new IllegalStateException(
          s"This pairwise translator has not been trained to translate from $srcLanguage to $tgtLanguage.")
      case Some(currentModel) => currentModel.infer(dataset)
    }
  }
}

object PairwiseTranslator {
  def apply(
      env: Environment,
      model: (Language, Vocabulary, Language, Vocabulary, Environment) => Model
  ): PairwiseTranslator = {
    new PairwiseTranslator(env, model)
  }
}
