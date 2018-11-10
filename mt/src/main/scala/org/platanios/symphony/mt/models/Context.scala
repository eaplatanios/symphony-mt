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

import org.platanios.symphony.mt.{Environment, Language}
import org.platanios.symphony.mt.data.DataConfig
import org.platanios.symphony.mt.models.parameters.ParameterManager
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api.Output
import org.platanios.tensorflow.api.learn.Mode

/**
  * @author Emmanouil Antonios Platanios
  */
case class Context(
    languages: Seq[(Language, Vocabulary)],
    env: Environment,
    parameterManager: ParameterManager,
    deviceManager: DeviceManager,
    dataConfig: DataConfig,
    modelConfig: ModelConfig,
    stage: Stage,
    mode: Mode,
    srcLanguageID: Output[Int],
    tgtLanguageID: Output[Int],
    tgtSequences: Option[Sequences[Int]]
) {
  def nextDevice(moveToNext: Boolean = true): String = {
    deviceManager.nextDevice(env, moveToNext)
  }
}
