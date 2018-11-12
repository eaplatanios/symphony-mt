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

package org.platanios.symphony.mt.experiments

import org.platanios.symphony.mt.{Environment, Language}
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.data.loaders._
import org.platanios.symphony.mt.experiments.config._
import org.platanios.symphony.mt.models.Model
import org.platanios.symphony.mt.models.parameters._
import org.platanios.symphony.mt.vocabulary.Vocabulary

import ch.qos.logback.classic.LoggerContext
import ch.qos.logback.classic.encoder.PatternLayoutEncoder
import ch.qos.logback.classic.spi.ILoggingEvent
import ch.qos.logback.core.FileAppender
import com.typesafe.config.{Config, ConfigFactory}
import org.slf4j.{Logger, LoggerFactory}

import java.nio.file.{Files, Path, Paths}

/**
  * @author Emmanouil Antonios Platanios
  */
class Experiment(val configFile: Path) {
  lazy val config: Config = ConfigFactory.parseFile(configFile.toFile).resolve()

  lazy val task: Experiment.Task = {
    config.get[String]("task") match {
      case "train" => Experiment.Train
      case "translate" => Experiment.Translate
      case "evaluate" => Experiment.Evaluate
      case value => throw new IllegalArgumentException(s"'$value' does not represent a valid task.")
    }
  }

  lazy val dataset: String = {
    config.get[String]("data.dataset")
  }

  lazy val (datasets, languages): (Seq[FileParallelDataset], Seq[(Language, Vocabulary)]) = {
    val languagePairs = Experiment.parseLanguagePairs(config.get[String]("training.languages"))
    val providedEvalLanguagePairs = Experiment.parseLanguagePairs(config.get[String]("evaluation.languages"))
    val evalLanguagePairs = if (providedEvalLanguagePairs.isEmpty) languagePairs else providedEvalLanguagePairs
    ParallelDatasetLoader.load(
      loaders = dataset match {
        case "iwslt14" => (languagePairs ++ evalLanguagePairs).toSeq.map(l => IWSLT14Loader(l._1, l._2, dataConfig))
        case "iwslt15" => (languagePairs ++ evalLanguagePairs).toSeq.map(l => IWSLT15Loader(l._1, l._2, dataConfig))
        case "iwslt16" => (languagePairs ++ evalLanguagePairs).toSeq.map(l => IWSLT16Loader(l._1, l._2, dataConfig))
        case "iwslt17" => (languagePairs ++ evalLanguagePairs).toSeq.map(l => IWSLT17Loader(l._1, l._2, dataConfig))
        case "wmt16" => (languagePairs ++ evalLanguagePairs).toSeq.map(l => WMT16Loader(l._1, l._2, dataConfig))
        case "ted_talks" => (languagePairs ++ evalLanguagePairs).toSeq.map(l => TEDTalksLoader(l._1, l._2, dataConfig))
      },
      workingDir = Some(environment.workingDir))
  }

  private lazy val parametersParser = {
    new ParametersParser(dataConfig)
  }

  lazy val environment: Environment = {
    new EnvironmentParser(toString).parse(config.get[Config]("environment"))
  }

  lazy val dataConfig: DataConfig = {
    DataConfigParser.parse(config.get[Config]("data"))
  }

  lazy val parameters: ParametersParser.Parameters = {
    parametersParser.parse(config.get[Config]("model.parameters"))
  }

  lazy val parameterManager: ParameterManager = {
    parameters.parameterManager
  }

  private lazy val modelParser = {
    // TODO: [EXPERIMENTS] Add support for other data types.
    new ModelParser[Float](
      task, dataset, datasets, languages, environment, parameterManager, dataConfig, "Model")
  }

  lazy val model: Model[_] = {
    modelParser.parse(config)
  }

  def initialize(): Unit = {
    val loggerContext = LoggerFactory.getILoggerFactory.asInstanceOf[LoggerContext]
    val patternLayoutEncoder = new PatternLayoutEncoder()
    patternLayoutEncoder.setPattern("%d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n")
    patternLayoutEncoder.setContext(loggerContext)
    patternLayoutEncoder.start()
    val fileAppender = new FileAppender[ILoggingEvent]()
    fileAppender.setFile(environment.workingDir.resolve("experiment.log").toAbsolutePath.toString)
    fileAppender.setEncoder(patternLayoutEncoder)
    fileAppender.setContext(loggerContext)
    fileAppender.start()
    LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME)
        .asInstanceOf[ch.qos.logback.classic.Logger]
        .addAppender(fileAppender)
  }

  def run(): Unit = {
    task match {
      case Experiment.Train => model.train(datasets.map(_.filterTypes(Train)))
      case Experiment.Translate => ???
      case Experiment.Evaluate => model.evaluate(model.evaluationConfig.datasets)
    }
  }

  protected def languagesStringHelper(languagePairs: Seq[(Language, Language)]): (String, Seq[String]) = {
    "Language Pairs" -> languagePairs.map(p => s"${p._1.abbreviation}-${p._2.abbreviation}").sorted
  }

  // def logSummary(): Unit = {
  //    Experiment.logger.info("Running an experiment with the following configuration:")
  //    val configTable = Seq(
  //      "Experiment" -> Seq("Type" -> {
  //        task match {
  //          case ExperimentConfig.Train => "Train"
  //          case ExperimentConfig.Translate => "Translate"
  //          case ExperimentConfig.Evaluate => "Evaluate"
  //        }
  //      }),
  //      "Dataset" -> Seq(
  //        "Name" -> """(\p{IsAlpha}+)(\p{IsDigit}+)""".r.replaceAllIn(dataset.map(_.toUpper), "$1-$2"),
  //        "Both Directions" -> trainBothDirections.toString,
  //        "Languages" -> providedLanguages,
  //        "Evaluation Languages" -> providedEvalLanguages,
  //        "Evaluation Tags" -> evalDatasetTags.mkString(", "),
  //        "Evaluation Metrics" -> evalMetrics.mkString(", ")),
  //      "Model" -> {
  //        Seq(
  //          "Architecture" -> modelArchitecture.toString,
  //          "Cell" -> {
  //            val parts = modelCell.split(":")
  //            s"${parts(0).toUpperCase()}[${parts(1)}]"
  //          },
  //          "Type" -> modelType.toString) ++ {
  //          if (modelType.isInstanceOf[HyperLanguage] || modelType.isInstanceOf[HyperLanguagePair])
  //            Seq("Language Embeddings Size" -> languageEmbeddingsSize.toString)
  //          else
  //            Seq.empty[(String, String)]
  //        } ++ Seq("Word Embeddings Size" -> wordEmbeddingsSize.toString) ++ {
  //          if (!modelArchitecture.isInstanceOf[GNMT])
  //            Seq(
  //              "Residual" -> residual.toString,
  //              "Attention" -> attention.toString)
  //          else
  //            Seq.empty[(String, String)]
  //        } ++ Seq(
  //          "Dropout" -> dropout.map(_.toString).getOrElse("Not Used"),
  //          "Label Smoothing" -> labelSmoothing.toString,
  //          "Identity Translations" -> trainUseIdentityTranslations.toString,
  //          "Beam Width" -> beamWidth.toString,
  //          "Length Penalty Weight" -> lengthPenaltyWeight.toString,
  //          "Decoding Max Length Factor" -> decoderMaxLengthFactor.toString,
  //          "" -> "", // This acts as a separator to help improve readability of the table.
  //          "Steps" -> numSteps.toString,
  //          "Summary Steps" -> summarySteps.toString,
  //          "Checkpoint Steps" -> checkpointSteps.toString,
  //          "" -> "", // This acts as a separator to help improve readability of the table.
  //          "Optimizer" -> {
  //            val parts = optString.split(":")
  //            s"${parts(0).capitalize}[lr=${parts(1)}]"
  //          },
  //          "Max Gradients Norm" -> optConfig.maxGradNorm.toString,
  //          "Colocate Gradients with Ops" -> optConfig.colocateGradientsWithOps.toString,
  //          "" -> "", // This acts as a separator to help improve readability of the table.
  //          "Log Loss Steps" -> logConfig.logLossSteps.toString,
  //          "Log Eval Steps" -> logConfig.logEvalSteps.toString,
  //          "Launch TensorBoard" -> logConfig.launchTensorBoard.toString,
  //          "TensorBoard Host" -> logConfig.tensorBoardConfig._1,
  //          "TensorBoard Port" -> logConfig.tensorBoardConfig._2.toString
  //        )
  //      },
  //      "Data Configuration" -> Seq(
  //        "Directory" -> dataConfig.dataDir.toString,
  //        "Loader Buffer Size" -> dataConfig.loaderBufferSize.toString,
  //        "Tokenizer" -> dataConfig.tokenizer.toString,
  //        "Cleaner" -> dataConfig.cleaner.toString,
  //        "Vocabulary" -> dataConfig.vocabulary.toString,
  //        "" -> "", // This acts as a separator to help improve readability of the table.
  //        "Percent Parallel" -> (dataConfig.parallelPortion * 100).toInt.toString,
  //        "Train Batch Size" -> dataConfig.trainBatchSize.toString,
  //        "Inference Batch Size" -> dataConfig.inferBatchSize.toString,
  //        "Evaluation Batch Size" -> dataConfig.evalBatchSize.toString,
  //        "Number of Buckets" -> dataConfig.numBuckets.toString,
  //        "Maximum Source Length" -> dataConfig.srcMaxLength.toString,
  //        "Maximum Target Length" -> dataConfig.tgtMaxLength.toString,
  //        "Prefetching Buffer Size" -> dataConfig.bufferSize.toString,
  //        "Number of Shards" -> dataConfig.numShards.toString,
  //        "Shard Index" -> dataConfig.shardIndex.toString,
  //        "TF - Number of Parallel Calls" -> dataConfig.numParallelCalls.toString,
  //        "" -> "", // This acts as a separator to help improve readability of the table.
  //        "Unknown Token" -> dataConfig.unknownToken,
  //        "Begin-of-Sequence Token" -> dataConfig.beginOfSequenceToken,
  //        "End-of-Sequence Token" -> dataConfig.endOfSequenceToken),
  //      "Environment" -> Seq(
  //        "Working Directory" -> env.workingDir.toString,
  //        "Number of GPUs" -> env.numGPUs.toString,
  //        "Random Seed" -> env.randomSeed.getOrElse("Not Set").toString,
  //        "TF - Trace Steps" -> env.traceSteps.map(_.toString).getOrElse("Not Set"),
  //        "TF - Allow Soft Placement" -> env.allowSoftPlacement.toString,
  //        "TF - Log Device Placement" -> env.logDevicePlacement.toString,
  //        "TF - Allow GPU Memory Growth" -> env.gpuAllowMemoryGrowth.toString,
  //        "TF - Use XLA" -> env.useXLA.toString,
  //        "TF - Parallel Iterations" -> env.parallelIterations.toString,
  //        "TF - Swap Memory" -> env.swapMemory.toString))
  //    ExperimentConfig.logTable(configTable, message => Experiment.logger.info(message))
  //  }

  override def toString: String = {
    val stringBuilder = new StringBuilder(s"$dataset")
    stringBuilder.append(s".${modelParser.tag(config, model).get}")
    stringBuilder.append(s".${parametersParser.tag(config.get[Config]("model.parameters"), parameters).get}")
    stringBuilder.append(s".${DataConfigParser.tag(config.get[Config]("data"), dataConfig).get}")
    stringBuilder.toString
  }
}

object Experiment {
  sealed trait Task
  case object Train extends Task
  case object Translate extends Task
  case object Evaluate extends Task

  private[experiments] def parseLanguagePairs(languages: String): Set[(Language, Language)] = {
    languages match {
      case l if l == "" => Set.empty[(Language, Language)]
      case l if l.contains(":") =>
        languages.split(',').map(p => {
          val parts = p.split(":")
          if (parts.length != 2)
            throw new IllegalArgumentException(s"'$p' is not a valid language pair.")
          (Language.fromAbbreviation(parts(0)), Language.fromAbbreviation(parts(1)))
        }).toSet
      case l =>
        l.split(',')
            .map(Language.fromAbbreviation)
            .combinations(2).map(p => (p(0), p(1)))
            .flatMap(p => Seq(p, (p._2, p._1)))
            .toSet
    }
  }

  private[Experiment] def logTable(table: Seq[(String, Seq[(String, String)])], logger: String => Unit): Unit = {
    val firstColumnWidth = Math.max(Math.max(table.map(_._1.length).max, table.flatMap(_._2.map(_._1.length)).max), 10)
    val secondColumnWidth = Math.max(table.flatMap(_._2.map(_._2.length)).max, 10)
    logger(s"╔═${"═" * firstColumnWidth}══${"═" * (secondColumnWidth + 2)}╗")
    table.zipWithIndex.foreach {
      case ((section, values), index) =>
        logger(s"║ %-${firstColumnWidth}s  ".format(section) + s"${" " * (secondColumnWidth + 2)}║")
        logger(s"╠═${"═" * firstColumnWidth}═╤${"═" * (secondColumnWidth + 2)}╣")
        values.foreach {
          case (key, value) if key != "" =>
            logger(s"║ %-${firstColumnWidth}s │ %-${secondColumnWidth}s ║".format(key, value))
          case _ => logger(s"╟─${"─" * firstColumnWidth}─┼${"─" * (secondColumnWidth + 2)}╢")
        }
        if (index < table.length - 1)
          logger(s"╠═${"═" * firstColumnWidth}═╧${"═" * (secondColumnWidth + 2)}╣")

    }
    logger(s"╚═${"═" * firstColumnWidth}═╧${"═" * (secondColumnWidth + 2)}╝")
  }

  @throws[IllegalArgumentException]
  def main(args: Array[String]): Unit = {
    val configFile = Paths.get(args(0))
    if (!Files.exists(configFile))
      throw new IllegalArgumentException(s"The provided configuration file ($configFile) does not exist.")
    val experiment = new Experiment(configFile)
    experiment.initialize()
    // experiment.logSummary()
    experiment.run()
  }
}
