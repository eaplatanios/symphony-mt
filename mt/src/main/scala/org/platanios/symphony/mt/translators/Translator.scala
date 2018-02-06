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

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.data.Dataset.MTTextLinesDataset
import org.platanios.symphony.mt.models.Model
import org.platanios.tensorflow.api.learn.StopCriteria

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Translator(val model: Model) {
  def train(
      trainDatasets: Seq[Translator.DatasetPair],
      devDatasets: Seq[Translator.DatasetPair] = null,
      testDatasets: Seq[Translator.DatasetPair] = null,
      stopCriteria: StopCriteria
  ): Unit

  def translate(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataset: MTTextLinesDataset
  ): MTTextLinesDataset
}

object Translator {
  case class DatasetPair(
      srcLanguage: Language,
      tgtLanguage: Language,
      srcDataset: MTTextLinesDataset,
      tgtDataset: MTTextLinesDataset)
}
