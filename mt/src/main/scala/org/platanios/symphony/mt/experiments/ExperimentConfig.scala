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

import org.platanios.symphony.mt.{Environment, Language, experiments}
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.data.loaders._
import org.platanios.symphony.mt.data.processors._
import org.platanios.symphony.mt.evaluation._
import org.platanios.symphony.mt.models._
import org.platanios.symphony.mt.vocabulary._
import org.platanios.tensorflow.api._

import ch.qos.logback.classic.LoggerContext
import ch.qos.logback.classic.encoder.PatternLayoutEncoder
import ch.qos.logback.classic.spi.ILoggingEvent
import ch.qos.logback.core.FileAppender
import org.slf4j.{Logger, LoggerFactory}

import java.io.File
import java.nio.file.{Path, Paths}

// TODO: Support more types of RNN cells.
// TODO: Make the optimizers more configurable (e.g., learning rate scheduling).

/**
  *
  * @param task
  * @param env
  * @param dataConfig
  * @param dataset
  * @param languagePairs
  * @param modelArchitecture
  * @param modelCell
  * @param modelType
  * @param languageEmbeddingsSize
  * @param wordEmbeddingsSize
  * @param residual
  * @param dropout
  * @param attention
  * @param labelSmoothing
  * @param beamWidth
  * @param lengthPenaltyWeight
  * @param decoderMaxLengthFactor
  * @param numSteps
  * @param summarySteps
  * @param checkpointSteps
  * @param optConfig
  * @param logConfig
  * @param evalDatasetTags
  *
  * @author Emmanouil Antonios Platanios
  */
case class ExperimentConfig(
    task: ExperimentConfig.Task = ExperimentConfig.Train,
    private val env: Environment = Environment(Paths.get(".")),
    dataConfig: DataConfig = DataConfig(),
    dataset: String = "",
    languagePairs: Seq[(Language, Language)] = Seq.empty,
    trainBackTranslation: Boolean = false,
    modelArchitecture: ModelArchitecture = BiRNN(),
    modelCell: String = "lstm:tanh",
    modelType: ModelType = Pairwise,
    languageEmbeddingsSize: Int = 16,
    wordEmbeddingsSize: Int = 256,
    residual: Boolean = false,
    dropout: Option[Float] = None,
    attention: Boolean = false,
    labelSmoothing: Float = 0.0f,
    beamWidth: Int = 10,
    lengthPenaltyWeight: Float = 0.0f,
    decoderMaxLengthFactor: Float = 2.0f,
    numSteps: Int = 100000,
    summarySteps: Int = 100,
    checkpointSteps: Int = 1000,
    optString: String = "gd:1.0",
    optConfig: Model.OptConfig = Model.OptConfig(),
    logConfig: Model.LogConfig = Model.LogConfig(),
    evalDatasetTags: Seq[String] = Seq.empty,
    evalMetrics: Seq[String] = Seq("bleu", "meteor", "hyp_len", "ref_len", "sen_cnt")
) {
  lazy val (datasets, languages) = {
    experiments.loadDatasets(dataset match {
      case "iwslt14" => languagePairs.map(l => IWSLT14Loader(l._1, l._2, dataConfig))
      case "iwslt15" => languagePairs.map(l => IWSLT15Loader(l._1, l._2, dataConfig))
      case "iwslt16" => languagePairs.map(l => IWSLT16Loader(l._1, l._2, dataConfig))
      case "wmt16" => languagePairs.map(l => WMT16Loader(l._1, l._2, dataConfig))
    }, Some(workingDir))
  }

  protected lazy val metrics: Seq[MTMetric] = evalMetrics.map(metric => {
    metric.split(":") match {
      case Array(name) if name == "bleu" => BLEU()(languages)
      case Array(name, maxOrder) if name == "bleu" => BLEU(maxOrder.toInt)(languages)
      case Array(name, maxOrder, smooth) if name == "bleu" => BLEU(maxOrder.toInt, smooth.toBoolean)(languages)
      case Array(name) if name == "meteor" => Meteor()(languages)
      case Array(name) if name == "ter" => TER()(languages)
      case Array(name) if name == "hyp_len" => SentenceLength(forHypothesis = true, name = "HypLen")
      case Array(name) if name == "ref_len" => SentenceLength(forHypothesis = false, name = "RefLen")
      case Array(name) if name == "sen_cnt" => SentenceCount(name = "#Sentences")
      case _ => throw new IllegalArgumentException(s"'$metric' does not represent a valid metric.")
    }
  })

  def workingDir: Path = env.workingDir.resolve(toString)

  def initialize(): Unit = {
    val loggerContext = LoggerFactory.getILoggerFactory.asInstanceOf[LoggerContext]
    val patternLayoutEncoder = new PatternLayoutEncoder()
    patternLayoutEncoder.setPattern("%d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n")
    patternLayoutEncoder.setContext(loggerContext)
    patternLayoutEncoder.start()
    val fileAppender = new FileAppender[ILoggingEvent]()
    fileAppender.setFile(workingDir.resolve("experiment.log").toAbsolutePath.toString)
    fileAppender.setEncoder(patternLayoutEncoder)
    fileAppender.setContext(loggerContext)
    fileAppender.start()
    LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME)
        .asInstanceOf[ch.qos.logback.classic.Logger]
        .addAppender(fileAppender)
  }

  def run(): Unit = {
    val env = this.env.copy(workingDir = workingDir)
    val parameterManager = modelType.getParametersManager(languageEmbeddingsSize, wordEmbeddingsSize)

    val evalDatasets = task match {
      case ExperimentConfig.Train | ExperimentConfig.Evaluate =>
        val evalTags = dataset match {
          case "iwslt14" => evalDatasetTags.map(tag => (s"IWSLT-14/$tag", IWSLT14Loader.Tag.fromName(tag)))
          case "iwslt15" => evalDatasetTags.map(tag => (s"IWSLT-15/$tag", IWSLT15Loader.Tag.fromName(tag)))
          case "iwslt16" => evalDatasetTags.map(tag => (s"IWSLT-16/$tag", IWSLT16Loader.Tag.fromName(tag)))
          case "wmt16" => evalDatasetTags.map(tag => (s"WMT-16/$tag", WMT16Loader.Tag.fromName(tag)))
        }
        evalTags.flatMap(t => datasets.map(d => (t._1, d.filterTags(t._2))))
      case ExperimentConfig.Translate => Seq.empty
    }

    val model = modelArchitecture.model(
      "Model", languages, dataConfig, env, parameterManager, trainBackTranslation,
      // Weird casting is necessary here to avoid compiling errors.
      modelCell, wordEmbeddingsSize, residual, dropout, attention, labelSmoothing,
      summarySteps, checkpointSteps, beamWidth, lengthPenaltyWeight, decoderMaxLengthFactor,
      optConfig, logConfig, evalDatasets, metrics)

    task match {
      case ExperimentConfig.Train =>
        model.train(datasets.map(_.filterTypes(Train)), tf.learn.StopCriteria.steps(numSteps))
      case ExperimentConfig.Translate => ???
      case ExperimentConfig.Evaluate => model.evaluate(evalDatasets)
    }
  }

  def logSummary(): Unit = {
    Experiment.logger.info("Running an experiment with the following configuration:")
    val configTable = Seq(
      "Experiment" -> Seq("Type" -> {
        task match {
          case ExperimentConfig.Train => "Train"
          case ExperimentConfig.Translate => "Translate"
          case ExperimentConfig.Evaluate => "Evaluate"
        }
      }),
      "Dataset" -> Seq(
        "Name" -> """(\p{IsAlpha}+)(\p{IsDigit}+)""".r.replaceAllIn(dataset.map(_.toUpper), "$1-$2"),
        "Language Pairs" -> languagePairs.map(p => s"${p._1.abbreviation}-${p._2.abbreviation}").mkString(", "),
        "Evaluation Tags" -> evalDatasetTags.mkString(", "),
        "Evaluation Metrics" -> evalMetrics.mkString(", ")),
      "Model" -> {
        Seq(
          "Architecture" -> modelArchitecture.toString,
          "Cell" -> {
            val parts = modelCell.split(":")
            s"${parts(0).toUpperCase()}[${parts(1)}]"
          },
          "Type" -> modelType.toString) ++ {
          if (modelType == HyperLanguage || modelType == HyperLanguagePair)
            Seq("Language Embeddings Size" -> languageEmbeddingsSize.toString)
          else
            Seq.empty[(String, String)]
        } ++ Seq("Word Embeddings Size" -> wordEmbeddingsSize.toString) ++ {
          if (!modelArchitecture.isInstanceOf[GNMT])
            Seq(
              "Residual" -> residual.toString,
              "Attention" -> attention.toString)
          else
            Seq.empty[(String, String)]
        } ++ Seq(
          "Dropout" -> dropout.map(_.toString).getOrElse("Not Used"),
          "Label Smoothing" -> labelSmoothing.toString,
          "Beam Width" -> beamWidth.toString,
          "Length Penalty Weight" -> lengthPenaltyWeight.toString,
          "Decoding Max Length Factor" -> decoderMaxLengthFactor.toString,
          "" -> "", // This acts as a separator to help improve readability of the table.
          "Steps" -> numSteps.toString,
          "Summary Steps" -> summarySteps.toString,
          "Checkpoint Steps" -> checkpointSteps.toString,
          "" -> "", // This acts as a separator to help improve readability of the table.
          "Optimizer" -> {
            val parts = optString.split(":")
            s"${parts(0).capitalize}[lr=${parts(1)}]"
          },
          "Max Gradients Norm" -> optConfig.maxGradNorm.toString,
          "Colocate Gradients with Ops" -> optConfig.colocateGradientsWithOps.toString,
          "" -> "", // This acts as a separator to help improve readability of the table.
          "Log Loss Steps" -> logConfig.logLossSteps.toString,
          "Log Eval Steps" -> logConfig.logEvalSteps.toString,
          "Launch TensorBoard" -> logConfig.launchTensorBoard.toString,
          "TensorBoard Host" -> logConfig.tensorBoardConfig._1,
          "TensorBoard Port" -> logConfig.tensorBoardConfig._2.toString
        )
      },
      "Data Configuration" -> Seq(
        "Directory" -> dataConfig.workingDir.toString,
        "Loader Buffer Size" -> dataConfig.loaderBufferSize.toString,
        "Tokenizer" -> dataConfig.tokenizer.toString,
        "Cleaner" -> dataConfig.cleaner.toString,
        "Vocabulary" -> dataConfig.vocabulary.toString,
        "" -> "", // This acts as a separator to help improve readability of the table.
        "Train Batch Size" -> dataConfig.trainBatchSize.toString,
        "Inference Batch Size" -> dataConfig.inferBatchSize.toString,
        "Evaluation Batch Size" -> dataConfig.evaluateBatchSize.toString,
        "Number of Buckets" -> dataConfig.numBuckets.toString,
        "Maximum Source Length" -> dataConfig.srcMaxLength.toString,
        "Maximum Target Length" -> dataConfig.tgtMaxLength.toString,
        "Prefetching Buffer Size" -> dataConfig.bufferSize.toString,
        "Number of Shards" -> dataConfig.numShards.toString,
        "Shard Index" -> dataConfig.shardIndex.toString,
        "TF - Number of Parallel Calls" -> dataConfig.numParallelCalls.toString,
        "" -> "", // This acts as a separator to help improve readability of the table.
        "Unknown Token" -> dataConfig.unknownToken,
        "Begin-of-Sequence Token" -> dataConfig.beginOfSequenceToken,
        "End-of-Sequence Token" -> dataConfig.endOfSequenceToken),
      "Environment" -> Seq(
        "Working Directory" -> env.workingDir.toString,
        "Number of GPUs" -> env.numGPUs.toString,
        "Random Seed" -> env.randomSeed.getOrElse("Not Set").toString,
        "TF - Allow Soft Placement" -> env.allowSoftPlacement.toString,
        "TF - Log Device Placement" -> env.logDevicePlacement.toString,
        "TF - Allow GPU Memory Growth" -> env.gpuAllowMemoryGrowth.toString,
        "TF - Use XLA" -> env.useXLA.toString,
        "TF - Parallel Iterations" -> env.parallelIterations.toString,
        "TF - Swap Memory" -> env.swapMemory.toString))
    ExperimentConfig.logTable(configTable, (message) => Experiment.logger.info(message))
  }

  override def toString: String = {
    val stringBuilder = new StringBuilder(s"$dataset")
    stringBuilder.append(s".${languagePairs.map(p => s"${p._1.abbreviation}-${p._2.abbreviation}").mkString(".")}")
    stringBuilder.append(s".$modelArchitecture")
    stringBuilder.append(s".$modelCell")
    stringBuilder.append(s".$modelType")
    if (modelType == HyperLanguage || modelType == HyperLanguagePair)
      stringBuilder.append(s".l:$languageEmbeddingsSize")
    stringBuilder.append(s".w:$wordEmbeddingsSize")
    if (!modelArchitecture.isInstanceOf[GNMT] && residual)
      stringBuilder.append(".r")
    if (!modelArchitecture.isInstanceOf[GNMT] && attention)
      stringBuilder.append(".a")
    dropout.map(d => stringBuilder.append(s".dropout:$d"))
    stringBuilder.append(s".ls:$labelSmoothing")
    stringBuilder.append(s".${dataConfig.tokenizer}")
    stringBuilder.append(s".${dataConfig.cleaner}")
    stringBuilder.append(s".${dataConfig.vocabulary}")
    stringBuilder.append(s".bs:${dataConfig.trainBatchSize}")
    stringBuilder.append(s".nb:${dataConfig.numBuckets}")
    stringBuilder.append(s".sml:${dataConfig.srcMaxLength}")
    stringBuilder.append(s".tml:${dataConfig.tgtMaxLength}")
    stringBuilder.toString
  }
}

object ExperimentConfig {
  private[ExperimentConfig] def logTable(table: Seq[(String, Seq[(String, String)])], logger: String => Unit): Unit = {
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

  implicit val floatRead: scopt.Read[Float] = scopt.Read.reads(_.toFloat)

  sealed trait Task
  case object Train extends Task
  case object Translate extends Task
  case object Evaluate extends Task

  implicit val taskRead: scopt.Read[Task] = {
    scopt.Read.reads {
      case "train" => Train
      case "translate" => Translate
      case "evaluate" => Evaluate
      case value => throw new IllegalArgumentException(s"'$value' does not represent a valid task.")
    }
  }

  implicit val tokenizerRead: scopt.Read[Tokenizer] = {
    scopt.Read.reads(value => {
      value.split(":") match {
        case Array(name) if name == "none" => NoTokenizer
        case Array(name) if name == "moses" => MosesTokenizer()
        case _ => throw new IllegalArgumentException(s"'$value' does not represent a valid tokenizer.")
      }
    })
  }

  implicit val cleanerRead: scopt.Read[Cleaner] = {
    scopt.Read.reads(value => {
      value.split(":") match {
        case Array(name) if name == "none" => NoCleaner
        case Array(name) if name == "moses" => MosesCleaner()
        case _ => throw new IllegalArgumentException(s"'$value' does not represent a valid cleaner.")
      }
    })
  }

  implicit val vocabularyRead: scopt.Read[DatasetVocabulary] = {
    scopt.Read.reads(value => {
      value.split(":") match {
        case Array(name) if name == "none" => NoVocabulary
        case Array(name) if name == "merged" => MergedVocabularies
        case Array(name) if name == "generated" => GeneratedVocabulary(SimpleVocabularyGenerator())
        case Array(name, sizeThreshold) if name == "generated" =>
          GeneratedVocabulary(SimpleVocabularyGenerator(sizeThreshold.toInt))
        case Array(name, sizeThreshold, countThreshold) if name == "generated" =>
          GeneratedVocabulary(SimpleVocabularyGenerator(sizeThreshold.toInt, countThreshold.toInt))
        case Array(name) if name == "bpe" => GeneratedVocabulary(BPEVocabularyGenerator())
        case Array(name, numMergeOps) if name == "bpe" => GeneratedVocabulary(BPEVocabularyGenerator(numMergeOps.toInt))
        case _ => throw new IllegalArgumentException(s"'$value' does not represent a valid vocabulary.")
      }
    })
  }

  implicit val optimizerRead: scopt.Read[tf.train.Optimizer] = {
    scopt.Read.reads(value => {
      value.split(":") match {
        case Array(name, learningRate) if name == "gd" =>
          tf.train.GradientDescent(learningRate.toDouble, learningRateSummaryTag = "LearningRate")
        case Array(name, learningRate) if name == "adadelta" =>
          tf.train.AdaDelta(learningRate.toDouble, learningRateSummaryTag = "LearningRate")
        case Array(name, learningRate) if name == "adagrad" =>
          tf.train.AdaGrad(learningRate.toDouble, learningRateSummaryTag = "LearningRate")
        case Array(name, learningRate) if name == "adam" =>
          tf.train.Adam(learningRate.toDouble, learningRateSummaryTag = "LearningRate")
        case Array(name, learningRate) if name == "lazy_adam" =>
          tf.train.LazyAdam(learningRate.toDouble, learningRateSummaryTag = "LearningRate")
        case Array(name, learningRate) if name == "amsgrad" =>
          tf.train.AMSGrad(learningRate.toDouble, learningRateSummaryTag = "LearningRate")
        case Array(name, learningRate) if name == "yf" =>
          tf.train.YellowFin(learningRate.toDouble, learningRateSummaryTag = "LearningRate")
        case _ => throw new IllegalArgumentException(s"'$value' does not represent a valid optimizer.")
      }
    })
  }

  val parser: scopt.OptionParser[ExperimentConfig] = new scopt.OptionParser[ExperimentConfig]("MT Experiment") {
    head("MT Experiment: A simple way to run MT experiments and reproduce experimental results.")

    opt[Task]('t', "task").required().valueName("<task>")
        .action((d, c) => c.copy(task = d))
        .text("Specifies the task to run.")

    opt[File]("working-dir").required().valueName("<file>")
        .action((d, c) => c.copy(env = c.env.copy(workingDir = d.toPath)))
        .text("Specifies the working directory to use for the experiment.")

    opt[File]("data-dir").required().valueName("<file>")
        .action((d, c) => c.copy(dataConfig = c.dataConfig.copy(workingDir = d.toPath)))
        .text("Specifies the directory to use for the data.")

    opt[String]("dataset").required().valueName("<name>")
        .action((d, c) => c.copy(dataset = d))
        .text("Specifies the dataset to use for the experiment. " +
            "Valid values are: 'iwslt14', 'iwslt15', 'iwslt16', and 'wmt16'.")

    opt[Seq[String]]("language-pairs").required().valueName("<srcLang1>:<tgtLang1>[,<srcLang2>:<tgtLang2>[...]]")
        .action((d, c) => c.copy(languagePairs = d.map(p => {
          val parts = p.split(":")
          if (parts.length != 2)
            throw new IllegalArgumentException(s"'$p' is not a valid language pair.")
          (Language.fromAbbreviation(parts(0)), Language.fromAbbreviation(parts(1)))
        })))
        .text("Specifies the language pairs to use for the experiment. Example value: 'en:vi,en:de'.")

    opt[Unit]("use-back-translations")
        .action((_, c) => c.copy(trainBackTranslation = true))
        .text("If used, back-translation data will be used while training " +
            "(i.e., translating back and forth from a single language.")

    opt[Seq[String]]("eval-datasets").valueName("<name1>[,<name2>[...]]")
        .action((d, c) => c.copy(evalDatasetTags = d))
        .text("Specifies the datasets to use for evaluation while training. Example value: 'tst2012,tst2013'.")

    opt[Seq[String]]("eval-metrics").valueName("<name1>[,<name2>[...]]")
        .action((d, c) => c.copy(evalMetrics = d))
        .text("Specifies the metrics to use for evaluation. Example value: 'bleu,hyp_len,ref_len,sen_cnt'.")

    opt[ModelArchitecture]("model-arch").required().valueName("<name>")
        .action((d, c) => c.copy(modelArchitecture = d))
        .text("Specifies the model name to use. " +
            "Valid values are: 'rnn:<num_enc_layers>:<num_dec_layers>', 'bi_rnn:<num_enc_layers>:<num_dec_layers>', " +
            "and 'gnmt:<num_bi_layers>:<num_uni_layers>:<num_uni_residual_layers>'.")

    opt[String]("model-cell").required().valueName("<name>")
        .action((d, c) => c.copy(modelCell = d))
        .text("Specifies the model cell to use. " +
            "Valid values are: 'gru[:<activation>]' and 'lstm[:<activation>[:<forget_bias>]]'.")

    opt[ModelType]("model-type").required().valueName("<type>")
        .action((d, c) => c.copy(modelType = d))
        .text("Specifies the model type to use. Valid values are: 'pair', 'hyper_lang', and 'hyper_lang_pair'.")

    opt[Int]("word-embed-size").valueName("<number>")
        .action((d, c) => c.copy(wordEmbeddingsSize = d))
        .text("Specifies the word embeddings size (for RNN models this is equal to the state size).")

    opt[Int]("lang-embed-size").valueName("<number>")
        .action((d, c) => c.copy(languageEmbeddingsSize = d))
        .text("Specifies the language embeddings size.")

    opt[Float]("dropout").valueName("<float>")
        .action((d, c) => c.copy(dropout = Some(d)))
        .text("Specifies the dropout probability for the model.")

    opt[Unit]("residual")
        .action((_, c) => c.copy(residual = true))
        .text("If used, residual connections will be added to the network. " +
            "Note that GNMT always uses residual connections.")

    opt[Unit]("attention")
        .action((_, c) => c.copy(attention = true))
        .text("If used, attention will be added to the network. Note that GNMT always uses attention.")

    opt[Float]("label-smoothing").valueName("<float>")
        .action((d, c) => c.copy(labelSmoothing = d))
        .text("Specifies the level of label smoothing to use.")

    opt[Int]("beam-width").valueName("<number>")
        .action((d, c) => c.copy(beamWidth = d))
        .text("Specifies the beam width to use while decoding.")

    opt[Float]("length-penalty").valueName("<float>")
        .action((d, c) => c.copy(lengthPenaltyWeight = d))
        .text("Specifies the length penalty weight to use while decoding.")

    opt[Float]("decoding-max-length-factor").valueName("<float>")
        .action((d, c) => c.copy(decoderMaxLengthFactor = d))
        .text("Specifies the maximum length factor to use while decoding " +
            "(i.e., maximum decoding length will be set to the input sequence length multiplied by this factor.")

    opt[Int]("num-steps").valueName("<number>")
        .action((d, c) => c.copy(numSteps = d))
        .text("Specifies the number of iterations to use for training.")

    opt[Int]("summary-steps").valueName("<number>")
        .action((d, c) => c.copy(summarySteps = d))
        .text("Specifies every how many steps to save summaries (e.g., for TensorBoard visualization.")

    opt[Int]("checkpoint-steps").valueName("<number>")
        .action((d, c) => c.copy(checkpointSteps = d))
        .text("Specifies every how many steps to save checkpoints.")

    opt[String]("opt").required().valueName("<optimizer>")
        .action((d, c) => c.copy(optString = d, optConfig = c.optConfig.copy(optimizer = implicitly[scopt.Read[tf.train.Optimizer]].reads(d))))
        .text("Specifies the optimizer to use while training. " +
            "Valid values are: 'gd:<learning_rate>', 'adadelta:<learning_rate>', 'adagrad:<learning_rate>', " +
            "'adam:<learning_rate>', 'lazy_adam:<learning_rate>', 'amsgrad:<learning_rate>', and 'yf:<learning_rate>'.")

    opt[Float]("opt-max-norm").required().valueName("<float>")
        .action((d, c) => c.copy(optConfig = c.optConfig.copy(maxGradNorm = d)))
        .text("Specifies the maximum gradient norm to use for global norm clipping while training.")

    opt[Unit]("opt-no-colocate-grads")
        .action((_, c) => c.copy(optConfig = c.optConfig.copy(colocateGradientsWithOps = false)))
        .text("If used, the gradients will not be colocated with their corresponding ops.")

    opt[Int]("log-loss-steps").valueName("<number>")
        .action((d, c) => c.copy(logConfig = c.logConfig.copy(logLossSteps = d)))
        .text("Specifies every how many steps to log the loss value.")

    opt[Int]("log-eval-steps").valueName("<number>")
        .action((d, c) => c.copy(logConfig = c.logConfig.copy(logEvalSteps = d)))
        .text("Specifies every how many steps to log the evaluation metric values.")

    opt[Unit]("launch-tensorboard")
        .action((_, c) => c.copy(logConfig = c.logConfig.copy(launchTensorBoard = true)))
        .text("If used, TensorBoard will be launched in the background while training.")

    opt[String]("tensorboard-host")
        .action((d, c) => c.copy(logConfig = c.logConfig.copy(tensorBoardConfig = (d, c.logConfig.tensorBoardConfig
            ._2))))
        .text("Specifies the host to use for TensorBoard.")

    opt[Int]("tensorboard-port")
        .action((d, c) => c.copy(logConfig = c.logConfig.copy(tensorBoardConfig = (c.logConfig.tensorBoardConfig._1,
            d))))
        .text("Specifies the port to use for TensorBoard.")

    opt[Int]("loader-buffer-size").valueName("<number>")
        .action((d, c) => c.copy(dataConfig = c.dataConfig.copy(loaderBufferSize = d)))
        .text("Specifies the buffer size to use when reading and writing files.")

    opt[Tokenizer]("tokenizer").valueName("<tokenizer>")
        .action((d, c) => c.copy(dataConfig = c.dataConfig.copy(tokenizer = d)))
        .text("Specifies the tokenizer to use for the text data. Valid values are: 'none' and 'moses'.")

    opt[Cleaner]("cleaner").valueName("<cleaner>")
        .action((d, c) => c.copy(dataConfig = c.dataConfig.copy(cleaner = d)))
        .text("Specifies the cleaner to use for the text data. Valid values are: 'none' and 'moses'.")

    opt[DatasetVocabulary]("vocabulary").valueName("<vocabulary>[:<option>...]")
        .action((d, c) => c.copy(dataConfig = c.dataConfig.copy(vocabulary = d)))
        .text("Specifies the vocabulary to use. " +
            "Valid values are: 'none', 'merged', 'generated[:size_threshold[:count_threshold]]', " +
            "and 'bpe[:num_merge_ops]'.")

    opt[Int]("batch-size").valueName("<number>")
        .action((d, c) => c.copy(dataConfig = c.dataConfig.copy(trainBatchSize = d)))
        .text("Specifies the batch size to use while training.")

    opt[Int]("infer-batch-size").valueName("<number>")
        .action((d, c) => c.copy(dataConfig = c.dataConfig.copy(inferBatchSize = d)))
        .text("Specifies the batch size to use while performing inference.")

    opt[Int]("eval-batch-size").valueName("<number>")
        .action((d, c) => c.copy(dataConfig = c.dataConfig.copy(evaluateBatchSize = d)))
        .text("Specifies the batch size to use while evaluating.")

    opt[Int]("num-buckets").valueName("<number>")
        .action((d, c) => c.copy(dataConfig = c.dataConfig.copy(numBuckets = d)))
        .text("Specifies the number of buckets to use when loading the training data.")

    opt[Int]("src-max-length").valueName("<number>")
        .action((d, c) => c.copy(dataConfig = c.dataConfig.copy(srcMaxLength = d)))
        .text("Specifies the maximum number of words to allow for source sentences while training.")

    opt[Int]("tgt-max-length").valueName("<number>")
        .action((d, c) => c.copy(dataConfig = c.dataConfig.copy(tgtMaxLength = d)))
        .text("Specifies the maximum number of words to allow for target sentences while training.")

    opt[Int]("buffer-size").valueName("<number>")
        .action((d, c) => c.copy(dataConfig = c.dataConfig.copy(bufferSize = d)))
        .text("Specifies the buffer size to use while prefetching training data.")

    opt[Int]("num-shards").valueName("<number>")
        .action((d, c) => c.copy(dataConfig = c.dataConfig.copy(numShards = d)))
        .text("Specifies the number of shards to use while loading the training data.")

    opt[Int]("shard-index").valueName("<number>")
        .action((d, c) => c.copy(dataConfig = c.dataConfig.copy(shardIndex = d)))
        .text("Specifies the shard index to use while loading the training data.")

    opt[Int]("loader-num-parallel").valueName("<number>")
        .action((d, c) => c.copy(dataConfig = c.dataConfig.copy(numParallelCalls = d)))
        .text("Specifies the number of parallel threads to allow for prefetching training data.")

    opt[String]("unknown-token").valueName("<word>")
        .action((d, c) => c.copy(dataConfig = c.dataConfig.copy(unknownToken = d)))
        .text("Specifies the token/word to use for representing unknown tokens.")

    opt[String]("begin-of-sequence-token").valueName("<word>")
        .action((d, c) => c.copy(dataConfig = c.dataConfig.copy(beginOfSequenceToken = d)))
        .text("Specifies the token/word to use for representing the beginning of a sequence/sentence.")

    opt[String]("end-of-sequence-token").valueName("<word>")
        .action((d, c) => c.copy(dataConfig = c.dataConfig.copy(endOfSequenceToken = d)))
        .text("Specifies the token/word to use for representing the end of a sequence/sentence.")

    opt[Unit]("disallow-soft-placement")
        .action((_, c) => c.copy(env = c.env.copy(allowSoftPlacement = false)))
        .text("If used, ops will not be allowed to be placed on devices other than the ones specified, if needed.")

    opt[Unit]("log-device-placement")
        .action((_, c) => c.copy(env = c.env.copy(logDevicePlacement = true)))
        .text("If used, a log will be produced containing the device assignment for each op.")

    opt[Unit]("gpu-allow-memory-growth")
        .action((_, c) => c.copy(env = c.env.copy(gpuAllowMemoryGrowth = true)))
        .text("If used, the GPU memory will not be allocated all at once, but rather on a need basis. " +
            "This reduces overall GPU memory usage but may cause a slowdown.")

    opt[Unit]("use-xla")
        .action((_, c) => c.copy(env = c.env.copy(useXLA = true)))
        .text("If used, the TensorFlow XLA compiler will be used. " +
            "Note that TensorFlow needs to first have been compiled with XLA support.")

    opt[Int]("num-gpus").valueName("<number>")
        .action((d, c) => c.copy(env = c.env.copy(numGPUs = d)))
        .text("Number of GPUs to use.")

    opt[Int]("parallel-iterations").valueName("<number>")
        .action((d, c) => c.copy(env = c.env.copy(parallelIterations = d)))
        .text("Number of parallel iterations to use in TensorFlow while loops.")

    opt[Unit]("no-swap-memory")
        .action((_, c) => c.copy(env = c.env.copy(swapMemory = false)))
        .text("If used, TensorFlow will not be allowed to swap things in and out GPU memory to save space.")

    opt[Int]("seed").valueName("<number>")
        .action((d, c) => c.copy(env = c.env.copy(randomSeed = Some(d))))
        .text("Random number generator seed to use.")

    help("help").text("Prints this usage information.")
  }
}
