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
import org.platanios.symphony.mt.data.scores.SentenceLength
import org.platanios.symphony.mt.experiments.Experiment
import org.platanios.symphony.mt.models.SentencePairsWithScores
import org.platanios.symphony.mt.models.curriculum.{DifficultyBasedCurriculum, SentencePairCurriculum}
import org.platanios.symphony.mt.models.curriculum.SentencePairCurriculum.{SourceSentenceScore, TargetSentenceScore}
import org.platanios.symphony.mt.models.curriculum.competency._
import org.platanios.tensorflow.api._

import com.typesafe.config.Config

import java.nio.file.Paths

/**
  * @author Emmanouil Antonios Platanios
  */
class TrainingConfigParser(
    dataset: String,
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
      summaryDir = Paths.get(config.get[String]("summary-dir")).resolve(dataset),
      checkpointSteps = config.get[Int]("checkpoint-frequency"),
      optimization = TrainingConfig.OptimizationConfig(
        optimizer = optimizer match {
          case "gd" => tf.train.GradientDescent(learningRate.get, learningRateSummaryTag = "LearningRate")
          case "adadelta" => tf.train.AdaDelta(learningRate.get, learningRateSummaryTag = "LearningRate")
          case "adafactor" => tf.train.Adafactor(learningRate, learningRateSummaryTag = "LearningRate")
          case "adagrad" => tf.train.AdaGrad(learningRate.get, learningRateSummaryTag = "LearningRate")
          case "rmsprop" => tf.train.RMSProp(learningRate.get, learningRateSummaryTag = "LearningRate")
          case "adam" =>
            val beta1 = config.get[Float]("optimization.beta1", 0.9f)
            val beta2 = config.get[Float]("optimization.beta2", 0.999f)
            tf.train.Adam(learningRate.get, beta1 = beta1, beta2 = beta2, learningRateSummaryTag = "LearningRate")
          case "lazy_adam" =>
            val beta1 = config.get[Float]("optimization.beta1", 0.9f)
            val beta2 = config.get[Float]("optimization.beta2", 0.999f)
            tf.train.LazyAdam(learningRate.get, beta1 = beta1, beta2 = beta2, learningRateSummaryTag = "LearningRate")
          case "amsgrad" =>
            val beta1 = config.get[Float]("optimization.beta1", 0.9f)
            val beta2 = config.get[Float]("optimization.beta2", 0.999f)
            tf.train.AMSGrad(learningRate.get, beta1 = beta1, beta2 = beta2, learningRateSummaryTag = "LearningRate")
          case "lazy_amsgrad" =>
            val beta1 = config.get[Float]("optimization.beta1", 0.9f)
            val beta2 = config.get[Float]("optimization.beta2", 0.999f)
            tf.train.LazyAMSGrad(learningRate.get, beta1 = beta1, beta2 = beta2, learningRateSummaryTag = "LearningRate")
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
      curriculum = config.getOption[Config]("curriculum").flatMap(parseCurriculum(_, languagePairs)))
  }

  @throws[IllegalArgumentException]
  private def parseCurriculum(
      curriculumConfig: Config,
      languagePairs: Set[(Language, Language)]
  ): Option[DifficultyBasedCurriculum[SentencePairsWithScores[String]]] = {
    curriculumConfig.get[String]("type") match {
      case "difficulty" =>
        val competency = parseCompetency(curriculumConfig.get[Config]("competency"))
        val score = curriculumConfig.get[String]("score") match {
          case "length" => SentenceLength
          case difficulty =>
            throw new IllegalArgumentException(s"'$difficulty' does not represent a valid difficulty type.")
        }
        val scoreSelectorString = curriculumConfig.get[String]("score-selector")
        val scoreSelector = scoreSelectorString match {
          case "source-sentence" => SourceSentenceScore
          case "target-sentence" => TargetSentenceScore
          case _ => throw new IllegalArgumentException(
            s"'$scoreSelectorString' does not represent a valid score selector.")
        }
        val maxNumHistogramBins = curriculumConfig.get[Int]("max-num-histogram-bins")
        Some(new SentencePairCurriculum(competency, score, scoreSelector, maxNumHistogramBins))
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
      case "exp-step" =>
        val initialValue = competencyConfig.get[Float]("initial-value")
        val numStepsToFullCompetency = competencyConfig.get[Float]("num-steps-full-competency")
        val power = competencyConfig.get[Int]("power")
        new ExponentialStepCompetency[Float](initialValue, numStepsToFullCompetency, power)
      case competencyType =>
        throw new IllegalArgumentException(s"'$competencyType' does not represent a valid competency type.")
    }
  }

  override def tag(config: Config, parsedValue: => TrainingConfig): Option[String] = {
    val bothDirections = config.get[Boolean]("both-directions")

    val stringBuilder = new StringBuilder()
    stringBuilder.append(s"bd:$bothDirections")
    stringBuilder.append(s".it:${bothDirections && config.get[Boolean]("use-identity-translations")}")
    stringBuilder.append(s".ls:${config.get[String]("label-smoothing")}")

    val optimizer = config.get[String]("optimization.optimizer")
    stringBuilder.append(s".opt:$optimizer")
    if (config.hasPath("optimization.learning-rate"))
      stringBuilder.append(s":${config.get[String]("optimization.learning-rate")}")
    optimizer match {
      case "adam" | "lazy_adam" | "amsgrad" | "lazy_amsgrad" =>
        stringBuilder.append(s":beta1:${config.get[String]("optimization.beta1", "0.9")}")
        stringBuilder.append(s":beta2:${config.get[String]("optimization.beta2", "0.999")}")
      case _ => ()
    }

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
          case "exp-step" =>
            val initialValue = config.get[String]("curriculum.competency.initial-value")
            val numStepsToFullCompetency = config.get[String]("curriculum.competency.num-steps-full-competency")
            val power = config.get[String]("curriculum.competency.power")
            stringBuilder.append(s":$initialValue:$numStepsToFullCompetency:$power")
          case _ => ()
        }
      }
    }

    Some(stringBuilder.toString)
  }
}
