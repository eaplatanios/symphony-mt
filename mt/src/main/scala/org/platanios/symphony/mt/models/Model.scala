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
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.evaluation
import org.platanios.symphony.mt.evaluation._
import org.platanios.symphony.mt.models.helpers.Common
import org.platanios.symphony.mt.models.hooks.TrainingLogger
import org.platanios.symphony.mt.models.parameters.ParameterManager
import org.platanios.symphony.mt.models.pivoting._
import org.platanios.symphony.mt.utilities.Encoding.tfStringToUTF8
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.config.{NoCheckpoints, TimeBasedCheckpoints}
import org.platanios.tensorflow.api.core.client.SessionConfig
import org.platanios.tensorflow.api.learn.{Mode, StopCriteria}
import org.platanios.tensorflow.api.learn.layers.{Input, Layer}
import org.platanios.tensorflow.api.learn.hooks.StepHookTrigger
import org.platanios.tensorflow.api.ops.training.optimizers.{GradientDescent, Optimizer}
import org.platanios.tensorflow.horovod._

import java.io.PrintWriter

import scala.io.Source

// TODO: Move embeddings initializer to the configuration.
// TODO: Add support for optimizer schedules (e.g., Adam for first 1000 steps and then SGD with a different learning rate.
// TODO: Customize hooks.

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Model[S] protected (
    val name: String,
    val languages: Seq[(Language, Vocabulary)],
    val dataConfig: DataConfig,
    val config: Model.Config,
    val optConfig: Model.OptConfig,
    val logConfig: Model.LogConfig = Model.LogConfig()
)(
    val evalDatasets: Seq[(String, FileParallelDataset, Float)] = Seq.empty,
    val evalMetrics: Seq[MTMetric] = Seq(
      BLEU()(languages),
      Meteor()(languages),
      TER()(languages),
      SentenceLength(forHypothesis = true, name = "HypLen"),
      SentenceLength(forHypothesis = false, name = "RefLen"),
      SentenceCount(name = "#Sentences"))
) {
  protected val languageIds: Map[Language, Int] = {
    val file = config.env.workingDir.resolve("languages.index").toFile
    if (file.exists()) {
      val indices = Source.fromFile(file).getLines().map(line => {
        val lineParts = line.split(',')
        (Language.fromName(lineParts(0)), lineParts(1).toInt)
      }).toMap
      if (!languages.forall(l => indices.contains(l._1)))
        throw new IllegalStateException(
          s"The existing language index file ($file) does not contain " +
              s"all of the provided languages (${languages.map(_._1.name).mkString(", ")}).")
      indices
    } else {
      val indices = languages.map(_._1).zipWithIndex.toMap
      val writer = new PrintWriter(file)
      indices.foreach(i => writer.write(s"${i._1.name},${i._2}\n"))
      writer.close()
      indices
    }
  }

  protected implicit val env             : Environment      = config.env
  protected implicit val parameterManager: ParameterManager = config.parameterManager
  protected implicit val deviceManager   : DeviceManager    = config.deviceManager

  /** Each input consists of a tuple containing:
    *   - The source language ID.
    *   - The target language ID.
    *   - A tensor containing a padded batch of sentences consisting of word IDs, in the source language.
    *   - A tensor containing the sentence lengths for the aforementioned padded batch.
    */
  protected val input      = Input[TFBatchWithLanguagesT, TFBatchWithLanguages, TFBatchWithLanguagesD, TFBatchWithLanguagesS]((INT32, INT32, STRING, INT32), (Shape(), Shape(), Shape(-1, -1), Shape(-1)))
  protected val trainInput = Input[TFBatchT, TFBatch, TFBatchD, TFBatchS]((STRING, INT32), (Shape(-1, -1), Shape(-1)))

  protected val estimator: tf.learn.InMemoryEstimator[
      TFBatchWithLanguagesT, TFBatchWithLanguages, TFBatchWithLanguagesD, TFBatchWithLanguagesS, TFBatchWithLanguage,
      (TFBatchWithLanguagesT, TFBatchT), (TFBatchWithLanguages, TFBatch),
      (TFBatchWithLanguagesD, TFBatchD), (TFBatchWithLanguagesS, TFBatchS),
      (TFBatchWithLanguage, TFBatch)] = {
    if (config.env.useHorovod)
      hvd.initialize()

    tf.createWith(
      nameScope = name,
      device = config.deviceManager.nextDevice(config.env, moveToNext = false)
    ) {
      var optimizer = optConfig.optimizer
      if (config.env.useHorovod)
        optimizer = hvd.DistributedOptimizer(optimizer)

      val model = optConfig.maxGradNorm match {
        case Some(norm) => learn.Model.supervised(
          input, inferLayer, trainLayer, trainInput, lossLayer, optimizer,
          clipGradients = tf.learn.ClipGradientsByGlobalNorm(norm),
          colocateGradientsWithOps = optConfig.colocateGradientsWithOps)
        case None => learn.Model.supervised(
          input, inferLayer, trainLayer, trainInput, lossLayer, optimizer,
          colocateGradientsWithOps = optConfig.colocateGradientsWithOps)
      }
      val summariesDir = config.env.workingDir.resolve("summaries")

      // Create estimator hooks.
      var hooks = Set[tf.learn.Hook]()

      // Add hooks (if distributed using Horovod, hooks are only added for the first process).
      if (!config.env.useHorovod || hvd.localRank == 0) {
        // Add logging hooks.
        if (logConfig.logLossSteps > 0) {
          val dataParallelFactor = if (config.env.useHorovod) hvd.size.toFloat else 1.0f
          hooks += TrainingLogger(
            log = true,
            trigger = StepHookTrigger(logConfig.logLossSteps),
            dataParallelFactor = dataParallelFactor)
        }
        if (logConfig.logEvalSteps > 0 && evalMetrics.nonEmpty) {
          val languagePairs = {
            if (config.evalLanguagePairs.nonEmpty) Some(config.evalLanguagePairs)
            else if (config.languagePairs.nonEmpty) Some(config.languagePairs)
            else None
          }
          val datasets = evalDatasets.filter(_._2.nonEmpty)
          if (datasets.nonEmpty) {
            val evalDatasets = Inputs.createEvalDatasets(dataConfig, config, datasets, languages, languagePairs)
            hooks += tf.learn.Evaluator(
              log = true, summariesDir, evalDatasets, evalMetrics, StepHookTrigger(logConfig.logEvalSteps),
              triggerAtEnd = true, numDecimalPoints = 6, name = "Evaluation")
          }
        }

        // Add summaries/checkpoints hooks.
        hooks ++= Set(
          tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = StepHookTrigger(100)),
          tf.learn.SummarySaver(summariesDir, StepHookTrigger(config.summarySteps)),
          tf.learn.CheckpointSaver(config.env.workingDir, StepHookTrigger(config.checkpointSteps)))

        env.traceSteps.foreach(numSteps =>
          hooks += tf.learn.TimelineHook(
            summariesDir, showDataFlow = true, showMemory = true, trigger = StepHookTrigger(numSteps)))

        // Add TensorBoard hook.
        if (logConfig.launchTensorBoard)
          hooks += tf.learn.TensorBoardHook(tf.learn.TensorBoardConfig(
            summariesDir, host = logConfig.tensorBoardConfig._1, port = logConfig.tensorBoardConfig._2))
      }

      if (config.env.useHorovod)
        hooks += hvd.BroadcastGlobalVariablesHook(0)

      var sessionConfig = SessionConfig(
        allowSoftPlacement = Some(config.env.allowSoftPlacement),
        logDevicePlacement = Some(config.env.logDevicePlacement),
        gpuAllowMemoryGrowth = Some(config.env.gpuAllowMemoryGrowth))
      if (config.env.useXLA)
        sessionConfig = sessionConfig.copy(optGlobalJITLevel = Some(SessionConfig.L1GraphOptimizerGlobalJIT))
      if (config.env.useHorovod)
        sessionConfig = sessionConfig.copy(gpuVisibleDevices = Some(Seq(hvd.localRank)))

      val checkpointConfig = {
        if (!config.env.useHorovod || hvd.localRank == 0)
          TimeBasedCheckpoints(600, 5, 10000)
        else
          NoCheckpoints
      }

      // Create estimator.
      tf.learn.InMemoryEstimator(
        model, tf.learn.Configuration(
          workingDir = Some(config.env.workingDir),
          sessionConfig = Some(sessionConfig),
          checkpointConfig = checkpointConfig,
          randomSeed = config.env.randomSeed),
        trainHooks = hooks)
    }
  }

  def train(datasets: Seq[FileParallelDataset], stopCriteria: StopCriteria): Unit = {
    val languagePairs = if (config.languagePairs.nonEmpty) Some(config.languagePairs) else None
    estimator.train(
      Inputs.createTrainDataset(
        dataConfig, config, datasets, languages, config.trainBackTranslation, repeat = true, isEval = false,
        languagePairs = languagePairs),
      stopCriteria)
  }

  def train(dataset: FileParallelDataset, stopCriteria: StopCriteria): Unit = {
    train(Seq(dataset), stopCriteria)
  }

  def translate(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataset: FileParallelDataset
  ): Iterator[(TFBatchWithLanguagesT, TFBatchWithLanguageT)] = {
    estimator.infer(Inputs.createInputDataset(dataConfig, config, dataset, srcLanguage, tgtLanguage, languages))
        .asInstanceOf[Iterator[(TFBatchWithLanguagesT, TFBatchWithLanguageT)]]
        .map(pair => {
          // TODO: We may be able to do this more efficiently.
          def decodeSequenceBatch(
              language: Tensor[DataType],
              sequences: Tensor[DataType],
              lengths: Tensor[DataType]
          ): (Tensor[DataType], Tensor[DataType]) = {
            val languageId = language.scalar.asInstanceOf[Int]
            val (unpackedSentences, unpackedLengths) = (sequences.unstack(), lengths.unstack())
            val decodedSentences = unpackedSentences.zip(unpackedLengths).map {
              case (s, len) =>
                val lenScalar = len.scalar.asInstanceOf[Int]
                val seq = s(0 :: lenScalar).entriesIterator.map(v => tfStringToUTF8(v.asInstanceOf[String])).toSeq
                languages(languageId)._2.decodeSequence(seq)
            }
            val decodedLengths = decodedSentences.map(_.length)
            val maxLength = decodedSentences.map(_.length).max
            val paddedDecodedSentences = decodedSentences.map(s => {
              s ++ Seq.fill(maxLength - s.length)(languages(languageId)._2.endOfSequenceToken)
            })
            (paddedDecodedSentences.toTensor, decodedLengths.toTensor)
          }

          val srcDecoded = decodeSequenceBatch(pair._1._1, pair._1._3, pair._1._4)
          val tgtDecoded = decodeSequenceBatch(pair._1._2, pair._2._2, pair._2._3)

          ((pair._1._1, pair._1._2, srcDecoded._1, srcDecoded._2), (pair._1._2, tgtDecoded._1, tgtDecoded._2))
        })
  }

//  def translate(
//      srcLanguage: (Language, Vocabulary),
//      tgtLanguage: (Language, Vocabulary),
//      input: (Tensor, Tensor)
//  ): Iterator[((Tensor, Tensor, Tensor, Tensor), (Tensor, Tensor, Tensor))] = {
//    TODO: Encode the input tensors.
//    translate(srcLanguage._1, tgtLanguage._1, TensorParallelDataset(
//      name = "TranslateTemp", vocabularies = Map(srcLanguage, tgtLanguage),
//      tensors = Map(srcLanguage._1 -> Seq(input))))
//  }

  def evaluate(
      datasets: Seq[(String, FileParallelDataset, Float)],
      metrics: Seq[MTMetric] = evalMetrics,
      maxSteps: Long = -1L,
      log: Boolean = true,
      saveSummaries: Boolean = true
  ): Seq[(String, Seq[(String, Tensor[FLOAT32])])] = {
    val rowNames = datasets.map(_._1)
    val firstColWidth = rowNames.map(_.length).max
    val colWidth = math.max(metrics.map(_.name.length).max, 10)
    if (log) {
      evaluation.logger.info(s"Evaluation results:")
      evaluation.logger.info(s"╔═${"═" * firstColWidth}═╤${metrics.map(_ => "═" * (colWidth + 2)).mkString("╤")}╗")
      evaluation.logger.info(s"║ ${" " * firstColWidth} │${metrics.map(s" %${colWidth}s ".format(_)).mkString("│")}║")
      evaluation.logger.info(s"╟─${"─" * firstColWidth}─┼${metrics.map(_ => "─" * (colWidth + 2)).mkString("┼")}╢")
    }
    val languagePairs = if (config.languagePairs.nonEmpty) Some(config.languagePairs) else None
    val evalDatasets = Inputs.createEvalDatasets(
      dataConfig, config, datasets.filter(_._2.nonEmpty), languages, languagePairs)
    val results = evalDatasets.map(dataset => {
      val values = estimator.evaluate(dataset._2, metrics, maxSteps, saveSummaries, name)
      if (log) {
        val line = s"║ %${firstColWidth}s │".format(dataset._1) + values.map(value => {
          if (value.shape.rank == 0 && value.dataType.isFloatingPoint) {
            val castedValue = value.cast(FLOAT32).scalar
            s" %$colWidth.4f ".format(castedValue)
          } else if (value.shape.rank == 0 && value.dataType.isInteger) {
            val castedValue = value.cast(INT64).scalar
            s" %${colWidth}d ".format(castedValue)
          } else {
            s" %${colWidth}s ".format("Not Scalar")
          }
        }).mkString("│") + "║"
        evaluation.logger.info(line)
      }
      dataset._1 -> metrics.map(_.name).zip(values)
    })
    if (log)
      evaluation.logger.info(s"╚═${"═" * firstColWidth}═╧${metrics.map(_ => "═" * (colWidth + 2)).mkString("╧")}╝")
    results
  }

  protected def trainLayer: Layer[(TFBatchWithLanguages, TFBatch), TFBatchWithLanguage] = {
    new Layer[(TFBatchWithLanguages, TFBatch), TFBatchWithLanguage](name) {
      override val layerType: String = "TrainLayer"

      override def forwardWithoutContext(
          input: (TFBatchWithLanguages, TFBatch)
      )(implicit mode: Mode): TFBatchWithLanguage = {
        tf.createWith(device = config.deviceManager.nextDevice(config.env, moveToNext = false)) {
          parameterManager.initialize(languages)
          parameterManager.setEnvironment(config.env)
          parameterManager.setDeviceManager(config.deviceManager)
          parameterManager.setContext((input._1._1, input._1._2))

          val srcSequence = mapToWordIds(input._1._1, input._1._3)
          val tgtSequence = mapToWordIds(input._1._2, input._2._1)
          val srcMapped = (input._1._1, input._1._2, srcSequence, input._1._4)
          val tgtMapped = (tgtSequence, input._2._2)
          val state = tf.variableScope("Encoder") {
            encoder(srcMapped)
          }
          val output = tf.variableScope("Decoder") {
            decoder(srcMapped, Some(tgtMapped), Some(state))
          }
          (input._1._2, output._1, output._2)
        }
      }
    }
  }

  protected def inferLayer: Layer[TFBatchWithLanguages, TFBatchWithLanguage] = {
    new Layer[TFBatchWithLanguages, TFBatchWithLanguage](name) {
      override val layerType: String = "InferLayer"

      override def forwardWithoutContext(
          input: TFBatchWithLanguages
      )(implicit mode: Mode): TFBatchWithLanguage = {
        tf.createWith(device = config.deviceManager.nextDevice(config.env, moveToNext = false)) {
          parameterManager.initialize(languages)
          parameterManager.setEnvironment(config.env)
          parameterManager.setDeviceManager(config.deviceManager)
          parameterManager.setContext((input._1, input._2))

          // TODO: We need to first have a check for whether the language pair is supported. If not supported, fallback multi-lingual translation method should be applied.

          val srcSequence = mapToWordIds(input._1, input._3)

          config.pivot match {
            case NoPivot =>
              val encoderInput = (input._1, input._2, srcSequence, input._4)
              val state = tf.variableScope("Encoder") {
                encoder(encoderInput)
              }
              val (tgtSentences, tgtLengths) = tf.variableScope("Decoder") {
                decoder(encoderInput, None, Some(state))
              }
              (input._2, tgtSentences, tgtLengths)
            case _ =>
              config.pivot.initialize(languages, parameterManager)
              val pivotingSequence = config.pivot.pivotingSequence(input._1, input._2)

              type LoopVariables = (Output, Output, Output, Output, Output)

              def predicateFn(loopVariables: LoopVariables): Output = {
                tf.less(loopVariables._1, tf.shape(loopVariables._5)(0))
              }

              def bodyFn(loopVariables: LoopVariables): LoopVariables = {
                val index = loopVariables._1
                val srcLanguage = loopVariables._2
                val srcSentences = loopVariables._3
                val srcLengths = loopVariables._4
                val language = tf.gather(loopVariables._5, index)
                val encoderInput = (srcLanguage, language, srcSentences, srcLengths)
                val state = tf.variableScope("Encoder") {
                  encoder(encoderInput)
                }
                val (tgtSentences, tgtLengths) = tf.variableScope("Decoder") {
                  decoder(encoderInput, None, Some(state))
                }
                (index + 1, language, tgtSentences, tgtLengths, loopVariables._5)
              }

              val results = tf.whileLoop[LoopVariables, (Shape, Shape, Shape, Shape, Shape)](
                predicateFn,
                bodyFn,
                loopVariables = (0, input._1, srcSequence, input._4, pivotingSequence),
                shapeInvariants = Some((Shape(), Shape(), Shape(-1, -1), Shape(-1), Shape(-1))))
              val tgtLanguage = results._2
              val tgtSentences = results._3
              val tgtLengths = results._4
              val decodedSequences = mapFromWordIds(tgtLanguage, tgtSentences)
              (tgtLanguage, decodedSequences, tgtLengths)
          }
        }
      }
    }
  }

  protected def lossLayer: Layer[(TFBatchWithLanguage, TFBatch), Output] = {
    new Layer[(TFBatchWithLanguage, TFBatch), Output](name) {
      override val layerType: String = "Loss"

      override def forwardWithoutContext(
          input: (TFBatchWithLanguage, TFBatch)
      )(implicit mode: Mode): Output = {
        tf.createWith(nameScope = "Loss", device = config.deviceManager.nextDevice(config.env, moveToNext = false)) {
          parameterManager.initialize(languages)
          parameterManager.setEnvironment(config.env)
          parameterManager.setDeviceManager(config.deviceManager)

          var tgtSequence = tf.createWithNameScope("TrainInputsToWordIDs") {
            mapToWordIds(input._1._1, input._2._1)
          }
          // TODO: Handle this shift more efficiently.
          // Shift the target sequence one step backward so the decoder is evaluated based using the correct previous
          // word used as input, rather than the previous predicted word.
          val tgtEosId = parameterManager
              .stringToIndexLookup(input._1._1)(tf.constant(dataConfig.endOfSequenceToken.toTensor))
              .cast(INT32)
          tgtSequence = tf.concatenate(Seq(
            tgtSequence, tf.fill(INT32, tf.stack(Seq(tf.shape(tgtSequence)(0), 1)))(tgtEosId)), axis = 1)
          val tgtSequenceLength = input._2._2 + 1
          val lossValue = loss(input._1._2, tgtSequence, tgtSequenceLength)
          tf.summary.scalar("Loss", lossValue)
          lossValue
        }
      }
    }
  }

  protected def mapToWordIds(language: Output, wordSequence: Output): Output = {
    parameterManager.stringToIndexLookup(language)(wordSequence).cast(INT32)
  }

  protected def mapFromWordIds(language: Output, wordIDSequence: Output): Output = {
    parameterManager.indexToStringLookup(language)(wordIDSequence.cast(INT64))
  }

  /**
    *
    * @param  input Tuple containing four tensors:
    *                 - `INT32` tensor containing the source language ID.
    *                 - `INT32` tensor containing the target language ID.
    *                 - `INT32` tensor with shape `[batchSize, inputLength]`, containing the sentence word IDs.
    *                 - `INT32` tensor with shape `[batchSize]`, containing the sentence lengths.
    * @param  mode  Current learning mode (e.g., training or evaluation).
    * @return   Tuple containing two tensors:
    *           - Encoder output, with shape `[batchSize, inputLength, hiddenSize]`.
    *           - Encoder-decoder attention bias and mask weights, with shape `[batchSize, inputLength]`.
    */
  protected def encoder(input: TFBatchWithLanguages)(implicit mode: Mode): S

  /**
    *
    * @return Tensor with shape `[batchSize, length, 1, hiddenSize]`.
    */
  protected def decoder(
      encoderInput: TFBatchWithLanguages,
      input: Option[TFBatch],
      state: Option[S]
  )(implicit mode: Mode): TFBatch

  protected def loss(predictedSequences: Output, targetSequences: Output, targetSequenceLengths: Output): Output = {
    val (lossSum, _) = Common.paddedCrossEntropy(
      predictedSequences, targetSequences, targetSequenceLengths, config.labelSmoothing, timeMajor = config.timeMajor)
    lossSum / tf.size(targetSequenceLengths).cast(FLOAT32)
  }
}

object Model {
  class Config protected (
      val env: Environment,
      val parameterManager: ParameterManager,
      val deviceManager: DeviceManager,
      val pivot: Pivot,
      val labelSmoothing: Float,
      val timeMajor: Boolean,
      val summarySteps: Int,
      val checkpointSteps: Int,
      val trainBackTranslation: Boolean,
      // The following is to allow training in one direction only (for a language pair).
      val languagePairs: Set[(Language, Language)],
      val evalLanguagePairs: Set[(Language, Language)])

  object Config {
    def apply(
        env: Environment,
        parameterManager: ParameterManager,
        deviceManager: DeviceManager = RoundRobinDeviceManager,
        pivot: Pivot = NoPivot,
        labelSmoothing: Float = 0.0f,
        timeMajor: Boolean = false,
        summarySteps: Int = 100,
        checkpointSteps: Int = 1000,
        trainBackTranslation: Boolean = false,
        languagePairs: Set[(Language, Language)] = Set.empty,
        evalLanguagePairs: Set[(Language, Language)] = Set.empty
    ): Config = {
      new Config(
        env, parameterManager, deviceManager, pivot, labelSmoothing, timeMajor, summarySteps, checkpointSteps,
        trainBackTranslation, languagePairs, evalLanguagePairs)
    }
  }

  case class OptConfig(
      maxGradNorm: Option[Float] = None,
      optimizer: Optimizer = GradientDescent(1.0f, learningRateSummaryTag = "LearningRate"),
      colocateGradientsWithOps: Boolean = true)

  case class LogConfig(
      logLossSteps: Int = 100,
      logEvalSteps: Int = 1000,
      launchTensorBoard: Boolean = false,
      tensorBoardConfig: (String, Int) = ("localhost", 6006))
}
