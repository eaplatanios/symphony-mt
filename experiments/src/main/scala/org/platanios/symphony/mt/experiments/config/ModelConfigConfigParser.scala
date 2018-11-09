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

package org.platanios.symphony.mt.experiments.config

import org.platanios.symphony.mt.experiments.Experiment
import org.platanios.symphony.mt.models.ModelConfig
import org.platanios.symphony.mt.models.ModelConfig.{InferenceConfig, LogConfig, OptConfig, TrainingConfig}
import org.platanios.symphony.mt.models.helpers.decoders.{GoogleLengthPenalty, NoLengthPenalty}
import org.platanios.symphony.mt.models.pivoting.NoPivot
import org.platanios.tensorflow.api.tf

import com.typesafe.config.Config

/**
  * @author Emmanouil Antonios Platanios
  */
object ModelConfigConfigParser extends ConfigParser[ModelConfig] {
  override def parse(config: Config): ModelConfig = {
    val bothDirections = config.getBoolean("both-directions")
    lazy val languagePairs = {
      val languagePairs = Experiment.parseLanguagePairs(config.getString("languages"))
      if (bothDirections)
        languagePairs.flatMap(p => Set(p, (p._2, p._1)))
      else
        languagePairs
    }

    lazy val evalLanguagePairs = {
      val providedPairs = Experiment.parseLanguagePairs(config.getString("eval-languages"))
      if (providedPairs.isEmpty) languagePairs else providedPairs
    }
    val useIdentityTranslations = {
      if (!bothDirections)
        false
      else
        config.getBoolean("training.use-identity-translations")
    }
    val optimizer = config.getString("training.optimization.optimizer")
    val learningRate = {
      if (config.hasPath("training.optimization.learning-rate"))
        Some(config.getDouble("training.optimization.learning-rate").toFloat)
      else
        None
    }
    ModelConfig(
      trainingConfig = TrainingConfig(
        useIdentityTranslations = useIdentityTranslations,
        labelSmoothing = config.getDouble("training.label-smoothing").toFloat,
        numSteps = config.getInt("training.num-steps"),
        summarySteps = config.getInt("training.summary-steps"),
        checkpointSteps = config.getInt("training.checkpoint-steps"),
        optConfig = OptConfig(
          optimizer = optimizer match {
            case "gd" => tf.train.GradientDescent(learningRate.get, learningRateSummaryTag = "LearningRate")
            case "adadelta" => tf.train.AdaDelta(learningRate.get, learningRateSummaryTag = "LearningRate")
            case "adafactor" => tf.train.Adafactor(learningRate, learningRateSummaryTag = "LearningRate")
            case "adagrad" => tf.train.AdaGrad(learningRate.get, learningRateSummaryTag = "LearningRate")
            case "rmsprop" => tf.train.RMSProp(learningRate.get, learningRateSummaryTag = "LearningRate")
            case "adam" => tf.train.Adam(learningRate.get, learningRateSummaryTag = "LearningRate")
            case "lazy_adam" => tf.train.LazyAdam(learningRate.get, learningRateSummaryTag = "LearningRate")
            case "amsgrad" => tf.train.AMSGrad(learningRate.get, learningRateSummaryTag = "LearningRate")
            case "lazy_amsgrad" => tf.train.LazyAMSGrad(learningRate.get, learningRateSummaryTag = "LearningRate")
            case "yellowfin" => tf.train.YellowFin(learningRate.get, learningRateSummaryTag = "LearningRate")
            case _ => throw new IllegalArgumentException(s"'$optimizer' does not represent a valid optimizer.")
          },
          maxGradNorm = {
            if (config.hasPath("training.optimization.max-grad-norm"))
              Some(config.getDouble("training.optimization.max-grad-norm").toFloat)
            else
              None
          },
          colocateGradientsWithOps = config.getBoolean("training.optimization.colocate-gradients-with-ops"))),
      inferenceConfig = InferenceConfig(
        pivot = NoPivot,
        beamWidth = config.getInt("inference.beam-width"),
        lengthPenalty = {
          if (config.hasPath("inference.length-penalty"))
            GoogleLengthPenalty(config.getDouble("inference.length-penalty").toFloat)
          else
            NoLengthPenalty
        },
        maxDecodingLengthFactor = config.getDouble("inference.max-decoding-length-factor").toFloat,
      ),
      logConfig = LogConfig(
        logLossSteps = config.getInt("training.logging.loss-steps"),
        logEvalSteps = config.getInt("training.logging.eval-steps"),
        launchTensorBoard = config.getBoolean("training.tensorboard.automatic-launch"),
        tensorBoardConfig = (
            config.getString("training.tensorboard.host"),
            config.getInt("training.tensorboard.port"))),
      timeMajor = config.getBoolean("time-major"),
      languagePairs = languagePairs,
      evalLanguagePairs = evalLanguagePairs)
  }

  override def tag(config: Config, parsedValue: => ModelConfig): Option[String] = {
    val bothDirections = config.getBoolean("both-directions")
    val providedLanguages = config.getString("languages")
    val optimizer = config.getString("training.optimization.optimizer")

    val stringBuilder = new StringBuilder()
    stringBuilder.append(s"${providedLanguages.replace(',', '.')}")
    stringBuilder.append(s".bd:$bothDirections")
    stringBuilder.append(s".it:${parsedValue.trainingConfig.useIdentityTranslations}")
    stringBuilder.append(s".ls:${parsedValue.trainingConfig.labelSmoothing}")

    if (config.hasPath("training.optimization.learning-rate"))
      stringBuilder.append(s".opt:$optimizer:${config.getDouble("training.optimization.learning-rate").toFloat}")
    else
      stringBuilder.append(s".opt:$optimizer")

    Some(stringBuilder.toString)
  }
}
