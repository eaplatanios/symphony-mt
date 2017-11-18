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

package org.platanios.symphony.mt.core

import org.platanios.symphony.mt.data.Datasets.MTTextLinesDataset
import org.platanios.tensorflow.api.learn.StopCriteria

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Translator(val configuration: Configuration = Configuration()) {
  def train(datasets: Seq[Translator.DatasetPair], stopCriteria: StopCriteria): Unit
  def translate(sourceLanguage: Language, targetLanguage: Language, dataset: MTTextLinesDataset): MTTextLinesDataset
}

object Translator {
  case class DatasetPair(
      sourceLanguage: Language,
      targetLanguage: Language,
      sourceDataset: MTTextLinesDataset,
      targetDataset: MTTextLinesDataset)
}
