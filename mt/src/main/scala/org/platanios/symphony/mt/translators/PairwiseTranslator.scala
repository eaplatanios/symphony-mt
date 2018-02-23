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
    override val model: (Language, Vocabulary, Language, Vocabulary, Environment) => Model[_]
) extends Translator(model) {
  protected val models: mutable.Map[(Language, Language), Model[_]] = mutable.Map.empty

  override def train(dataset: ParallelDataset, stopCriteria: StopCriteria): Unit = {
    dataset.languagePairs(includeReversed = true).foreach {
      case (srcLanguage, tgtLanguage) =>
        val currentModel = models.getOrElseUpdate(
          (srcLanguage, tgtLanguage),
          model(
            srcLanguage, dataset.vocabulary(srcLanguage), tgtLanguage, dataset.vocabulary(tgtLanguage),
            env.copy(workingDir = env.workingDir.resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"))))
        currentModel.train(() => dataset.filterTypes(Train).toTFBilingual(srcLanguage, tgtLanguage, repeat = true), stopCriteria)
    }
  }

  override def translate(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataset: ParallelDataset
  ): Iterator[((Tensor, Tensor), (Tensor, Tensor))] = {
    val currentModel = models.getOrElseUpdate(
      (srcLanguage, tgtLanguage),
      model(
        srcLanguage, dataset.vocabulary(srcLanguage), tgtLanguage, dataset.vocabulary(tgtLanguage),
        env.copy(workingDir = env.workingDir.resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"))))
    currentModel.infer(() => dataset.toTFMonolingual(srcLanguage))
  }
}

object PairwiseTranslator {
  def apply(
      env: Environment,
      model: (Language, Vocabulary, Language, Vocabulary, Environment) => Model[_]
  ): PairwiseTranslator = {
    new PairwiseTranslator(env, model)
  }
}
