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

package org.platanios.symphony.mt.models

import org.platanios.symphony.mt.{Environment, Language, LogConfig}
import org.platanios.symphony.mt.data.Datasets.MTTextLinesDataset
import org.platanios.symphony.mt.data.{DataConfig, Vocabulary}
import org.platanios.tensorflow.api.learn.StopCriteria

// TODO: Move embeddings initializer to the configuration.
// TODO: Add support for optimizer schedules (e.g., Adam for first 1000 steps and then SGD with a different learning rate.
// TODO: Customize evaluation metrics, hooks, etc.

/**
  * @author Emmanouil Antonios Platanios
  */
trait Model {
  val name: String = "Model"

  val env        : Environment = Environment()
  val dataConfig : DataConfig  = DataConfig()
  val trainConfig: TrainConfig = TrainConfig()
  val inferConfig: InferConfig = InferConfig()
  val logConfig  : LogConfig   = LogConfig()

  val srcLanguage  : Language
  val tgtLanguage  : Language
  val srcVocabulary: Vocabulary
  val tgtVocabulary: Vocabulary

  val srcTrainDataset: MTTextLinesDataset = null
  val tgtTrainDataset: MTTextLinesDataset = null
  val srcDevDataset  : MTTextLinesDataset = null
  val tgtDevDataset  : MTTextLinesDataset = null
  val srcTestDataset : MTTextLinesDataset = null
  val tgtTestDataset : MTTextLinesDataset = null

  def train(
      srcTrainDataset: MTTextLinesDataset,
      tgtTrainDataset: MTTextLinesDataset,
      stopCriteria: StopCriteria = StopCriteria(Some(trainConfig.numSteps))
  ): Unit
}