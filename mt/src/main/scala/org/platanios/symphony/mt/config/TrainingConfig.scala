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

package org.platanios.symphony.mt.config

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.models.SentencePairs
import org.platanios.symphony.mt.models.curriculum.Curriculum
import org.platanios.tensorflow.api.ops.training.optimizers.{GradientDescent, Optimizer}

/**
  * @author Emmanouil Antonios Platanios
  */
case class TrainingConfig(
    languagePairs: Set[(Language, Language)],
    useIdentityTranslations: Boolean,
    labelSmoothing: Float,
    numSteps: Int,
    summarySteps: Int,
    checkpointSteps: Int,
    optimization: TrainingConfig.OptimizationConfig,
    logging: TrainingConfig.LoggingConfig,
    curriculum: Curriculum[SentencePairs[String]])

object TrainingConfig {
  case class OptimizationConfig(
      optimizer: Optimizer = GradientDescent(1.0f, learningRateSummaryTag = "LearningRate"),
      maxGradNorm: Option[Float] = None,
      colocateGradientsWithOps: Boolean = true)

  case class LoggingConfig(
      logLossFrequency: Int = 100,
      launchTensorBoard: Boolean = false,
      tensorBoardConfig: (String, Int) = ("localhost", 6006))
}
