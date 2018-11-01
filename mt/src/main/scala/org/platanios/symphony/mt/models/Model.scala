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
import org.platanios.symphony.mt.models.Model.DecodingMode
import org.platanios.symphony.mt.models.helpers.Common
import org.platanios.symphony.mt.models.hooks.TrainingLogger
import org.platanios.symphony.mt.models.parameters.ParameterManager
import org.platanios.symphony.mt.models.pivoting._
import org.platanios.symphony.mt.utilities.Encoding.tfStringToUTF8
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.config.TimeBasedCheckpoints
import org.platanios.tensorflow.api.core.client.SessionConfig
import org.platanios.tensorflow.api.learn.{Mode, StopCriteria}
import org.platanios.tensorflow.api.learn.layers.{Input, Layer}
import org.platanios.tensorflow.api.learn.hooks.StepHookTrigger
import org.platanios.tensorflow.api.ops.training.optimizers.{GradientDescent, Optimizer}

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
  protected val input: Input[SentencesWithLanguagePair[String]] = {
    Input(
      dataType = (INT32, INT32, (STRING, INT32)),
      shape = (Shape(), Shape(), (Shape(-1, -1), Shape(-1))),
      name = "Input")
  }

  protected val trainInput: Input[Sentences[String]] = {
    Input(
      dataType = (STRING, INT32),
      shape = (Shape(-1, -1), Shape(-1)),
      name = "TrainInput")
  }

  protected val estimator: TranslationEstimator = {
    tf.createWith(
      nameScope = name,
      device = config.deviceManager.nextDevice(config.env, moveToNext = false)
    ) {
      var optimizer = optConfig.optimizer
      val model = optConfig.maxGradNorm match {
        case Some(norm) =>
          tf.learn.Model.supervised(
            input = input,
            trainInput = trainInput,
            layer = inferLayer,
            trainLayer = trainLayer,
            loss = lossLayer,
            optimizer = optimizer,
            clipGradients = tf.learn.ClipGradientsByGlobalNorm(norm),
            colocateGradientsWithOps = optConfig.colocateGradientsWithOps)
        case None =>
          tf.learn.Model.supervised(
            input = input,
            trainInput = trainInput,
            layer = inferLayer,
            trainLayer = trainLayer,
            loss = lossLayer,
            optimizer = optimizer,
            colocateGradientsWithOps = optConfig.colocateGradientsWithOps)
      }
      val summariesDir = config.env.workingDir.resolve("summaries")

      // Create estimator hooks.
      var hooks = Set[tf.learn.Hook]()

      // Add logging hooks.
      if (logConfig.logLossSteps > 0) {
        hooks += TrainingLogger(
          log = true,
          trigger = StepHookTrigger(logConfig.logLossSteps))
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

      var sessionConfig = SessionConfig(
        allowSoftPlacement = Some(config.env.allowSoftPlacement),
        logDevicePlacement = Some(config.env.logDevicePlacement),
        gpuAllowMemoryGrowth = Some(config.env.gpuAllowMemoryGrowth))
      if (config.env.useXLA)
        sessionConfig = sessionConfig.copy(optGlobalJITLevel = Some(SessionConfig.L1GraphOptimizerGlobalJIT))

      // Create estimator.
      tf.learn.InMemoryEstimator(
        model, tf.learn.Configuration(
          workingDir = Some(config.env.workingDir),
          sessionConfig = Some(sessionConfig),
          checkpointConfig = TimeBasedCheckpoints(600, 5, 10000),
          randomSeed = config.env.randomSeed),
        trainHooks = hooks)
    }
  }

  def train(datasets: Seq[FileParallelDataset], stopCriteria: StopCriteria): Unit = {
    val languagePairs = if (config.languagePairs.nonEmpty) Some(config.languagePairs) else None
    estimator.train(
      data = Inputs.createTrainDataset(
        dataConfig = dataConfig,
        modelConfig = config,
        datasets = datasets,
        languages = languages,
        includeIdentityTranslations = config.trainIdentityTranslations,
        repeat = true,
        isEval = false,
        languagePairs = languagePairs),
      stopCriteria = stopCriteria)
  }

  def train(dataset: FileParallelDataset, stopCriteria: StopCriteria): Unit = {
    train(Seq(dataset), stopCriteria)
  }

  def translate(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataset: FileParallelDataset
  ): Iterator[(SentencesWithLanguagePairValue, SentencesWithLanguageValue)] = {
    val inputDataset = Inputs.createInputDataset(
      dataConfig = dataConfig,
      modelConfig = config,
      dataset = dataset,
      srcLanguage = srcLanguage,
      tgtLanguage = tgtLanguage,
      languages = languages)
    estimator.infer(inputDataset)
        .asInstanceOf[Iterator[(SentencesWithLanguagePairValue, SentencesWithLanguageValue)]]
        .map(pair => {
          // TODO: We may be able to do this more efficiently.
          def decodeSequenceBatch(
              language: Tensor[Int],
              sequences: Tensor[String],
              lengths: Tensor[Int]
          ): (Tensor[String], Tensor[Int]) = {
            val languageId = language.scalar
            val (unpackedSentences, unpackedLengths) = (sequences.unstack(), lengths.unstack())
            val decodedSentences = unpackedSentences.zip(unpackedLengths).map {
              case (s, len) =>
                val lenScalar = len.scalar
                val seq = s(0 :: lenScalar).entriesIterator.map(v => tfStringToUTF8(v)).toSeq
                languages(languageId)._2.decodeSequence(seq)
            }
            val decodedLengths = decodedSentences.map(_.length)
            val maxLength = decodedSentences.map(_.length).max
            val paddedDecodedSentences = decodedSentences.map(s => {
              s ++ Seq.fill(maxLength - s.length)(languages(languageId)._2.endOfSequenceToken)
            })
            (paddedDecodedSentences: Tensor[String], decodedLengths: Tensor[Int])
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
  ): Seq[(String, Seq[(String, Tensor[Float])])] = {
    // Create the evaluation datasets that may only consider a subset of the language pairs.
    val languagePairs = if (config.languagePairs.nonEmpty) Some(config.languagePairs) else None
    val evalDatasets = Inputs.createEvalDatasets(
      dataConfig = dataConfig,
      modelConfig = config,
      datasets = datasets.filter(_._2.nonEmpty),
      languages = languages,
      languagePairs = languagePairs)

    // Log the header of the evaluation results table.
    val rowNames = datasets.map(_._1)
    val firstColWidth = rowNames.map(_.length).max
    val colWidth = math.max(metrics.map(_.name.length).max, 10)
    if (log) {
      evaluation.logger.info(s"Evaluation results:")
      evaluation.logger.info(s"╔═${"═" * firstColWidth}═╤${metrics.map(_ => "═" * (colWidth + 2)).mkString("╤")}╗")
      evaluation.logger.info(s"║ ${" " * firstColWidth} │${metrics.map(s" %${colWidth}s ".format(_)).mkString("│")}║")
      evaluation.logger.info(s"╟─${"─" * firstColWidth}─┼${metrics.map(_ => "─" * (colWidth + 2)).mkString("┼")}╢")
    }

    // Run the actual evaluation and log the results.
    val results = evalDatasets.map(dataset => {
      val values = estimator.evaluate(dataset._2, metrics, maxSteps, saveSummaries, name)
      if (log) {
        val line = s"║ %${firstColWidth}s │".format(dataset._1) + values.map(value => {
          if (value.shape.rank == 0 && value.dataType.isFloatingPoint) {
            val castedValue = value.toFloat.scalar
            s" %$colWidth.4f ".format(castedValue)
          } else if (value.shape.rank == 0 && value.dataType.isInteger) {
            val castedValue = value.toLong.scalar
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

  protected def trainLayer: Layer[(SentencesWithLanguagePair[String], Sentences[String]), SentencesWithLanguage[Float]] = {
    new Layer[(SentencesWithLanguagePair[String], Sentences[String]), SentencesWithLanguage[Float]](name) {
      override val layerType: String = "TrainLayer"

      override def forwardWithoutContext(
          input: (SentencesWithLanguagePair[String], Sentences[String])
      )(implicit mode: Mode): SentencesWithLanguage[Float] = {
        tf.createWith(device = config.deviceManager.nextDevice(config.env, moveToNext = false)) {
          parameterManager.initialize(languages)
          parameterManager.setEnvironment(config.env)
          parameterManager.setDeviceManager(config.deviceManager)

          val srcLanguage = input._1._1
          val tgtLanguage = input._1._2

          implicit val context: Output[Int] = {
            tf.stack(Seq(srcLanguage, tgtLanguage))
          }

          // Map words to word indices in the vocabulary.
          val srcSentencesMapped = mapToWordIds(srcLanguage, /* source sentences */ input._1._3._1)
          val tgtSentencesMapped = mapToWordIds(tgtLanguage, /* target sentences */ input._2._1)
          val srcMapped = (srcLanguage, tgtLanguage, (srcSentencesMapped, /* source sentence lengths */ input._1._3._2))
          val tgtMapped = (tgtSentencesMapped, /* target sentence lengths */ input._2._2)

          // Encode the source sentences.
          val state = tf.variableScope("Encoder") {
            encoder(srcMapped)
          }

          // Decode to obtain the target sentences.
          val decodedTgtSentences = tf.variableScope("Decoder") {
            decoder(Model.DecodingTrainMode, srcMapped, Some(tgtMapped), Some(state))
          }

          (tgtLanguage, decodedTgtSentences)
        }
      }
    }
  }

  protected def inferLayer: Layer[SentencesWithLanguagePair[String], SentencesWithLanguage[String]] = {
    new Layer[SentencesWithLanguagePair[String], SentencesWithLanguage[String]](name) {
      override val layerType: String = "InferLayer"

      override def forwardWithoutContext(
          input: SentencesWithLanguagePair[String]
      )(implicit mode: Mode): SentencesWithLanguage[String] = {
        tf.createWith(device = config.deviceManager.nextDevice(config.env, moveToNext = false)) {
          parameterManager.initialize(languages)
          parameterManager.setEnvironment(config.env)
          parameterManager.setDeviceManager(config.deviceManager)

          val srcLanguage = input._1
          val tgtLanguage = input._2

          implicit val context: Output[Int] = tf.stack(Seq(srcLanguage, tgtLanguage))

          // Map words to word to indices in the vocabulary.
          val srcSentencesMapped = mapToWordIds(srcLanguage, /* source sentences */ input._3._1)
          val srcMapped = (srcLanguage, tgtLanguage, (srcSentencesMapped, /* source sentence lengths */ input._3._2))

          // Perform inference based on the current pivoting strategy for multi-lingual translations.
          config.pivot match {
            case NoPivot =>
              // Encode the source sentences.
              val encoderInput = srcMapped
              val state = tf.variableScope("Encoder") {
                encoder(encoderInput)
              }

              // Decode to obtain the target sentences.
              val (tgtSentences, tgtSentenceLengths) = tf.variableScope("Decoder") {
                decoder(Model.DecodingInferMode, encoderInput, None, Some(state))
              }

              // Map from word indices in the vocabulary, back to actual words and return the result.
              val decodedSequences = mapFromWordIds(tgtLanguage, tgtSentences)
              (tgtLanguage, (decodedSequences, tgtSentenceLengths))
            case _ =>
              config.pivot.initialize(languages, parameterManager)

              // Construct a pivoting sequence over languages and loop over it using a while loop.
              val pivotingSequence = config.pivot.pivotingSequence(srcLanguage, tgtLanguage)

              type LoopVariables = (Output[Int], Output[Int], Sentences[Int])

              def predicateFn(loopVariables: LoopVariables): Output[Boolean] = {
                tf.less(loopVariables._1, tf.shape(pivotingSequence).slice(0).toInt)
              }

              def bodyFn(loopVariables: LoopVariables): LoopVariables = {
                val index = loopVariables._1

                // Obtain the source and target language in this step of the pivoting chain.
                val currentSrcLanguage = loopVariables._2
                val currentTgtLanguage = tf.gather(pivotingSequence, index)

                implicit val context: Output[Int] = {
                  tf.stack(Seq(currentSrcLanguage, currentTgtLanguage))
                }

                // Encode the source sentences in this step of the pivoting chain.
                val encoderInput = (currentSrcLanguage, currentTgtLanguage, /* source sentences */ loopVariables._3)
                val state = tf.variableScope("Encoder") {
                  encoder(encoderInput)
                }

                // Decode to obtain the target sentences in this step of the pivoting chain.
                val tgtSentences = tf.variableScope("Decoder") {
                  decoder(Model.DecodingInferMode, encoderInput, None, Some(state))
                }

                (index + 1, currentTgtLanguage, tgtSentences)
              }

              val srcSequences = srcMapped._3._1
              val srcSequenceLengths = srcMapped._3._2

              // Set the shape invariants for the while loop.
              srcSequences.setShape(Shape(-1, -1))
              srcSequenceLengths.setShape(Shape(-1))

              val results = tf.whileLoop(
                predicateFn, bodyFn,
                loopVariables = (Output.constant[Int](0), srcLanguage, (srcSequences, srcSequenceLengths)))
              val decodedSequences = mapFromWordIds(tgtLanguage, /* target sentences */ results._3._1)
              (tgtLanguage, (decodedSequences, /* target sentence lengths */ results._3._2))
          }
        }
      }
    }
  }

  protected def lossLayer: Layer[(SentencesWithLanguage[Float], (SentencesWithLanguagePair[String], Sentences[String])), Output[Float]] = {
    new Layer[(SentencesWithLanguage[Float], (SentencesWithLanguagePair[String], Sentences[String])), Output[Float]](name) {
      override val layerType: String = "Loss"

      override def forwardWithoutContext(
          input: (SentencesWithLanguage[Float], (SentencesWithLanguagePair[String], Sentences[String]))
      )(implicit mode: Mode): Output[Float] = {
        tf.createWith(nameScope = "Loss", device = config.deviceManager.nextDevice(config.env, moveToNext = false)) {
          parameterManager.initialize(languages)
          parameterManager.setEnvironment(config.env)
          parameterManager.setDeviceManager(config.deviceManager)

          val tgtLanguage = input._1._1

          var tgtSequences = tf.nameScope("TrainInputsToWordIDs") {
            mapToWordIds(tgtLanguage, /* reference target sentences */ input._2._2._1)
          }

          // TODO: Handle this shift more efficiently.
          // Shift the target sequence one step backward so the decoder is evaluated based using the correct previous
          // word used as input, rather than the previous predicted word.
          val tgtEosId = parameterManager
              .stringToIndexLookup(tgtLanguage)(Output.constant[String](dataConfig.endOfSequenceToken)).toInt
          tgtSequences = tf.concatenate(Seq(
            tgtSequences,
            tf.fill(tf.stack[Int](Seq(tf.shape(tgtSequences).slice(0).toInt, 1)))(tgtEosId)
          ), axis = 1)
          val tgtSequenceLengths = input._2._2._2 + 1
          val lossValue = loss(/* predicted target sentences */ input._1._2._1, tgtSequences, tgtSequenceLengths)
          tf.summary.scalar("Loss", lossValue)
          lossValue
        }
      }
    }
  }

  protected def mapToWordIds(
      language: Output[Int],
      wordSequence: Output[String]
  ): Output[Int] = {
    parameterManager.stringToIndexLookup(language)(wordSequence)
  }

  protected def mapFromWordIds(
      language: Output[Int],
      wordIDSequence: Output[Int]
  ): Output[String] = {
    parameterManager.indexToStringLookup(language)(wordIDSequence)
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
  protected def encoder(input: SentencesWithLanguagePair[Int])(implicit
      mode: Mode,
      context: Output[Int]
  ): S

  /**
    *
    * @return Tensor with shape `[batchSize, length, 1, hiddenSize]`.
    */
  protected def decoder[O: TF](
      decodingMode: DecodingMode[O],
      encoderInput: SentencesWithLanguagePair[Int],
      input: Option[Sentences[Int]],
      state: Option[S]
  )(implicit
      mode: Mode,
      context: Output[Int]
  ): Sentences[O]

  protected def loss(
      predictedSequences: Output[Float],
      tgtSequences: Output[Int],
      tgtSequenceLengths: Output[Int]
  ): Output[Float] = {
    val (lossSum, _) = Common.paddedCrossEntropy(
      logits = predictedSequences,
      labels = tgtSequences,
      labelLengths = tgtSequenceLengths,
      labelSmoothing = config.labelSmoothing,
      timeMajor = config.timeMajor)
    lossSum / tf.size(tgtSequenceLengths).toFloat
  }
}

object Model {
  sealed trait DecodingMode[O]
  case object DecodingTrainMode extends DecodingMode[Float]
  case object DecodingInferMode extends DecodingMode[Int]

  class Config protected(
      val env: Environment,
      val parameterManager: ParameterManager,
      val deviceManager: DeviceManager,
      val pivot: Pivot,
      val labelSmoothing: Float,
      val timeMajor: Boolean,
      val summarySteps: Int,
      val checkpointSteps: Int,
      val trainIdentityTranslations: Boolean,
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
