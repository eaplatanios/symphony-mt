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
import org.platanios.symphony.mt.data.{LoadedDataset, MTInferDataset}
import org.platanios.symphony.mt.models.Model
import org.platanios.symphony.mt.translators.actors.SystemConfig
import org.platanios.symphony.mt.translators.actors.Messages.SystemTrainRequest
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api.Tensor
import org.platanios.tensorflow.api.learn.StopCriteria

import akka.actor._

/**
  * @author Emmanouil Antonios Platanios
  */
class SymphonyTranslator protected (
    val systemConfig: SystemConfig,
    override val model: (Language, Vocabulary, Language, Vocabulary, Environment) => Model,
    val name: String
) extends Translator(model) {
  protected val actorSystem: ActorSystem = ActorSystem(s"SymphonyTranslator$name")
  protected val system     : ActorRef    = {
    actorSystem.actorOf(actors.System.props(systemConfig, model), s"System$name")
  }

  override def train(dataset: LoadedDataset, stopCriteria: StopCriteria): Unit = {
    system ! SystemTrainRequest(dataset)
  }

  @throws[IllegalStateException]
  override def translate(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataset: () => MTInferDataset
  ): Iterator[((Tensor, Tensor), (Tensor, Tensor))] = ???
}

object SymphonyTranslator {
  def apply(
      systemConfig: SystemConfig,
      model: (Language, Vocabulary, Language, Vocabulary, Environment) => Model,
      name: String
  ): SymphonyTranslator = {
    new SymphonyTranslator(systemConfig, model, name)
  }
}
