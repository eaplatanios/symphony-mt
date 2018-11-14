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

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.config.TrainingConfig
import org.platanios.symphony.mt.data.{DataConfig, FileParallelDataset}
import org.platanios.symphony.mt.experiments.Experiment
import org.platanios.symphony.mt.models.SentencePairs
import org.platanios.symphony.mt.models.curriculum.{Curriculum, DifficultyBasedCurriculum}
import org.platanios.symphony.mt.models.curriculum.competency.{Competency, LinearStepCompetency, SquareRootStepCompetency}
import org.platanios.symphony.mt.models.curriculum.difficulty.{AdaptiveLengthBasedDifficulty, Difficulty}
import org.platanios.symphony.mt.models.curriculum.difficulty.LengthBasedDifficulty.{SourceLengthSelector, TargetLengthSelector}
import org.platanios.tensorflow.api._

import com.typesafe.config.Config

/**
  * @author Emmanouil Antonios Platanios
  */
class TrainingConfigParser(
    datasets: => Seq[FileParallelDataset],
    dataConfig: => DataConfig
) extends ConfigParser[TrainingConfig] {
  @throws[IllegalArgumentException]
  override def parse(config: Config): TrainingConfig = {
    val bothDirections = config.get[Boolean]("both-directions")
    val languagePairs = {
      val languagePairs = Experiment.parseLanguagePairs(config.get[String]("languages"))
      if (bothDirections)
        languagePairs.flatMap(p => Set(p, (p._2, p._1)))
      else
        languagePairs
    }
    val optimizer = config.get[String]("optimization.optimizer")
    val learningRate = config.getOption[Float]("optimization.learning-rate")
    TrainingConfig(
      languagePairs = languagePairs,
      useIdentityTranslations = bothDirections && config.get[Boolean]("use-identity-translations"),
      cacheData = config.get[Boolean]("cache-data"),
      labelSmoothing = config.get[Float]("label-smoothing"),
      numSteps = config.get[Int]("num-steps"),
      summarySteps = config.get[Int]("summary-frequency"),
      checkpointSteps = config.get[Int]("checkpoint-frequency"),
      optimization = TrainingConfig.OptimizationConfig(
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
        maxGradNorm = config.getOption[Float]("optimization.max-grad-norm"),
        colocateGradientsWithOps = config.get[Boolean]("optimization.colocate-gradients-with-ops")),
      logging = TrainingConfig.LoggingConfig(
        logLossFrequency = config.get[Int]("logging.log-loss-frequency"),
        launchTensorBoard = config.get[Boolean]("tensorboard.automatic-launch"),
        tensorBoardConfig = (
            config.get[String]("tensorboard.host"),
            config.get[Int]("tensorboard.port"))),
      curriculum =
          config.getOption[Config]("curriculum").map(parseCurriculum(_, languagePairs))
              .getOrElse(Curriculum.none[SentencePairs[String]]))
  }

  @throws[IllegalArgumentException]
  private def parseCurriculum(
      curriculumConfig: Config,
      languagePairs: Set[(Language, Language)]
  ): Curriculum[SentencePairs[String]] = {
    curriculumConfig.get[String]("type") match {
      case "difficulty" =>
        new DifficultyBasedCurriculum(
          difficulty = parseDifficulty(curriculumConfig.get[Config]("difficulty"), languagePairs),
          competency = parseCompetency(curriculumConfig.get[Config]("competency")))
      case curriculumType =>
        throw new IllegalArgumentException(s"'$curriculumType' does not represent a valid curriculum type.")
    }
  }

  @throws[IllegalArgumentException]
  private def parseCompetency(competencyConfig: Config): Competency[Output[Float]] = {
    competencyConfig.get[String]("type") match {
      case "linear-step" =>
        val initialValue = competencyConfig.get[Float]("initial-value")
        val numStepsToFullCompetency = competencyConfig.get[Float]("num-steps-full-competency")
        new LinearStepCompetency[Float](initialValue, numStepsToFullCompetency)
      case "sqrt-step" =>
        val initialValue = competencyConfig.get[Float]("initial-value")
        val numStepsToFullCompetency = competencyConfig.get[Float]("num-steps-full-competency")
        new SquareRootStepCompetency[Float](initialValue, numStepsToFullCompetency)
      case competencyType =>
        throw new IllegalArgumentException(s"'$competencyType' does not represent a valid competency type.")
    }
  }

  @throws[IllegalArgumentException]
  private def parseDifficulty(
      difficultyConfig: Config,
      languagePairs: Set[(Language, Language)]
  ): Difficulty[SentencePairs[String]] = {
    difficultyConfig.get[String]("type") match {
      case "adaptive-length" =>
        val lengthSelectorString = difficultyConfig.get[String]("selector")
        val lengthSelector = lengthSelectorString match {
          case "source-sentence-length" => SourceLengthSelector
          case "target-sentence-length" => TargetLengthSelector
          case _ => throw new IllegalArgumentException(
            s"'$lengthSelectorString' does not represent a valid length selector.")
        }
        new AdaptiveLengthBasedDifficulty[String](lengthSelector, datasets, languagePairs.toSeq, dataConfig)
      case competencyType =>
        throw new IllegalArgumentException(s"'$competencyType' does not represent a valid competency type.")
    }
  }

  override def tag(config: Config, parsedValue: => TrainingConfig): Option[String] = {
    val bothDirections = config.get[Boolean]("both-directions")

    val stringBuilder = new StringBuilder()
    stringBuilder.append(s"bd:$bothDirections")
    stringBuilder.append(s".it:${bothDirections && config.get[Boolean]("use-identity-translations")}")
    stringBuilder.append(s".ls:${config.get[Float]("label-smoothing")}")
    stringBuilder.append(s".opt:${config.get[String]("optimization.optimizer")}")
    if (config.hasPath("optimization.learning-rate"))
      stringBuilder.append(s"${config.get[Float]("optimization.learning-rate")}")

    if (config.hasPath("curriculum.type")) {
      val curriculumType = config.get[String]("curriculum.type")
      stringBuilder.append(s".curr:$curriculumType")
      if (config.hasPath("curriculum.competency.type")) {
        val competencyType = config.get[String]("curriculum.competency.type")
        stringBuilder.append(s".comp:$competencyType")
        competencyType match {
          case "linear-step" =>
            val initialValue = config.get[String]("curriculum.competency.initial-value")
            val numStepsToFullCompetency = config.get[String]("curriculum.competency.num-steps-full-competency")
            stringBuilder.append(s":$initialValue:$numStepsToFullCompetency")
          case "sqrt-step" =>
            val initialValue = config.get[String]("curriculum.competency.initial-value")
            val numStepsToFullCompetency = config.get[String]("curriculum.competency.num-steps-full-competency")
            stringBuilder.append(s":$initialValue:$numStepsToFullCompetency")
          case _ => ()
        }
      }
    }

    Some(stringBuilder.toString)
  }
}
