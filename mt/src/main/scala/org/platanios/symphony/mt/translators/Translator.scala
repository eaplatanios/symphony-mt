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
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.StopCriteria

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Translator(val model: (Language, Vocabulary, Language, Vocabulary, Environment) => Model) {
  def train(dataset: LoadedDataset, stopCriteria: StopCriteria): Unit

  @throws[IllegalStateException]
  def translate(
      srcLang: Language,
      tgtLang: Language,
      dataset: () => MTInferDataset
  ): Iterator[((Tensor, Tensor), (Tensor, Tensor))]

  @throws[IllegalStateException]
  def translate(srcLang: Language, tgtLang: Language, input: (Tensor, Tensor)): (Tensor, Tensor) = {
    translate(srcLang, tgtLang, () => tf.data.TensorDataset(input).asInstanceOf[MTInferDataset]).next()._2
  }
}
