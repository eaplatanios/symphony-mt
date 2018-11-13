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
import org.platanios.symphony.mt.config.{EvaluationConfig, InferenceConfig, TrainingConfig}
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
import org.platanios.tensorflow.api.core.client.SessionConfig
import org.platanios.tensorflow.api.learn.{Counter, Mode, StopCriteria}
import org.platanios.tensorflow.api.learn.layers.{Input, Layer}
import org.platanios.tensorflow.api.learn.hooks.StepHookTrigger

import java.io.PrintWriter

import scala.io.Source

// TODO: Move embeddings initializer to the configuration.
// TODO: Add support for optimizer schedules (e.g., Adam for first 1000 steps and then SGD with a different learning rate.
// TODO: Customize hooks.

/**
  * @author Emmanouil Antonios Platanios
  */
class Model[Code](
    val name: String,
    val encoder: Transformation.Encoder[Code],
    val decoder: Transformation.Decoder[Code],
    val languages: Seq[(Language, Vocabulary)],
    val env: Environment,
    val parameterManager: ParameterManager,
    val dataConfig: DataConfig,
    val trainingConfig: TrainingConfig,
    val inferenceConfig: InferenceConfig,
    val evaluationConfig: EvaluationConfig,
    val deviceManager: DeviceManager = RoundRobinDeviceManager
) {
  implicit val context: Context = Context(
    languages = languages,
    env = env,
    parameterManager = parameterManager,
    dataConfig = dataConfig,
    trainingConfig = trainingConfig,
    inferenceConfig = inferenceConfig,
    evaluationConfig = evaluationConfig,
    deviceManager = deviceManager)

  protected val languageIds: Map[Language, Int] = {
    val file = env.workingDir.resolve("languages.index").toFile
    if (file.exists()) {
      val indices = Source.fromFile(file).getLines().zipWithIndex.map(line => {
        (Language.fromName(line._1), line._2)
      }).toMap
      if (!languages.forall(l => indices.contains(l._1)))
        throw new IllegalStateException(
          s"The existing language index file ($file) does not contain " +
              s"all of the provided languages (${languages.map(_._1.name).mkString(", ")}).")
      indices
    } else {
      val writer = new PrintWriter(file)
      val indices = languages.map(l => {
        val language = l._1
        writer.write(s"${language.name}\n")
        language
      }).zipWithIndex.toMap
      writer.close()
      indices
    }
  }

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
      device = deviceManager.nextDevice(env, moveToNext = false)
    ) {
      val model = trainingConfig.optimization.maxGradNorm match {
        case Some(norm) =>
          tf.learn.Model.supervised(
            input = input,
            trainInput = trainInput,
            layer = inferLayer,
            trainLayer = trainLayer,
            loss = lossLayer,
            optimizer = trainingConfig.optimization.optimizer,
            clipGradients = tf.learn.ClipGradientsByGlobalNorm(norm),
            colocateGradientsWithOps = trainingConfig.optimization.colocateGradientsWithOps)
        case None =>
          tf.learn.Model.supervised(
            input = input,
            trainInput = trainInput,
            layer = inferLayer,
            trainLayer = trainLayer,
            loss = lossLayer,
            optimizer = trainingConfig.optimization.optimizer,
            colocateGradientsWithOps = trainingConfig.optimization.colocateGradientsWithOps)
      }
      val summariesDir = env.workingDir.resolve("summaries")

      // Create estimator hooks.
      var hooks = Set[tf.learn.Hook]()

      // Add logging hooks.
      if (trainingConfig.logging.logLossFrequency > 0) {
        hooks += TrainingLogger(
          log = true,
          trigger = StepHookTrigger(trainingConfig.logging.logLossFrequency))
      }
      if (evaluationConfig.frequency > 0 && evaluationConfig.metrics.nonEmpty) {
        val languagePairs = {
          if (evaluationConfig.languagePairs.nonEmpty) Some(evaluationConfig.languagePairs)
          else if (trainingConfig.languagePairs.nonEmpty) Some(trainingConfig.languagePairs)
          else None
        }
        val datasets = evaluationConfig.datasets.filter(_._2.nonEmpty)
        if (datasets.nonEmpty) {
          val evalDatasets = Inputs.createEvalDatasets(dataConfig, trainingConfig, datasets, languages, languagePairs)
          hooks += tf.learn.Evaluator(
            log = true, summariesDir, evalDatasets, evaluationConfig.metrics, StepHookTrigger(
              numSteps = evaluationConfig.frequency,
              startStep = evaluationConfig.frequency),
            triggerAtEnd = true, numDecimalPoints = 6, name = "Evaluation")
        }
      }

      // Add summaries/checkpoints hooks.
      val checkpointSteps = math.min(trainingConfig.checkpointSteps, evaluationConfig.frequency)
      hooks ++= Set(
        tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = StepHookTrigger(100)),
        tf.learn.SummarySaver(summariesDir, StepHookTrigger(trainingConfig.summarySteps)),
        tf.learn.CheckpointSaver(env.workingDir, StepHookTrigger(checkpointSteps)))

      env.traceSteps.foreach(numSteps =>
        hooks += tf.learn.TimelineHook(
          summariesDir, showDataFlow = true, showMemory = true, trigger = StepHookTrigger(numSteps)))

      // Add TensorBoard hook.
      if (trainingConfig.logging.launchTensorBoard) {
        hooks += tf.learn.TensorBoardHook(tf.learn.TensorBoardConfig(
          summariesDir,
          host = trainingConfig.logging.tensorBoardConfig._1,
          port = trainingConfig.logging.tensorBoardConfig._2))
      }

      var sessionConfig = SessionConfig(
        allowSoftPlacement = Some(env.allowSoftPlacement),
        logDevicePlacement = Some(env.logDevicePlacement),
        gpuAllowMemoryGrowth = Some(env.gpuAllowMemoryGrowth))
      if (env.useXLA)
        sessionConfig = sessionConfig.copy(optGlobalJITLevel = Some(SessionConfig.L1GraphOptimizerGlobalJIT))

      // Create estimator.
      tf.learn.InMemoryEstimator(
        model, tf.learn.Configuration(
          workingDir = Some(env.workingDir),
          sessionConfig = Some(sessionConfig),
          randomSeed = env.randomSeed),
        trainHooks = hooks)
    }
  }

  def train(datasets: Seq[FileParallelDataset], stopCriteria: Option[StopCriteria] = None): Unit = {
    val languagePairs = if (trainingConfig.languagePairs.nonEmpty) Some(trainingConfig.languagePairs) else None
    estimator.train(
      data = Inputs.createTrainDataset(
        dataConfig = dataConfig,
        trainingConfig = trainingConfig,
        datasets = datasets,
        languages = languages,
        includeIdentityTranslations = trainingConfig.useIdentityTranslations,
        cache = trainingConfig.cacheData,
        repeat = true,
        isEval = false,
        languagePairs = languagePairs),
      stopCriteria = stopCriteria.getOrElse(StopCriteria(Some(trainingConfig.numSteps))))
  }

  def translate(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataset: FileParallelDataset
  ): Iterator[(SentencesWithLanguagePairValue, SentencesWithLanguageValue)] = {
    val inputDataset = Inputs.createInputDataset(
      dataConfig = dataConfig,
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
      datasets: Seq[(String, FileParallelDataset, Float)] = evaluationConfig.datasets,
      metrics: Seq[MTMetric] = evaluationConfig.metrics,
      maxSteps: Long = -1L,
      log: Boolean = true,
      saveSummaries: Boolean = true
  ): Seq[(String, Seq[(String, Tensor[Float])])] = {
    // Create the evaluation datasets that may only consider a subset of the language pairs.
    val languagePairs = if (trainingConfig.languagePairs.nonEmpty) Some(trainingConfig.languagePairs) else None
    val evalDatasets = Inputs.createEvalDatasets(
      dataConfig = dataConfig,
      trainingConfig = trainingConfig,
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
        trainingConfig.curriculum.initialize()
        tf.createWith(device = deviceManager.nextDevice(env, moveToNext = false)) {
          parameterManager.initialize(languages, trainingConfig)

          val srcLanguageID = input._1._1
          val tgtLanguageID = input._1._2

          // Map tokens to token indices in the vocabulary.
          val mappedSrcSequences = Sequences(
            sequences = parameterManager.stringToIndexLookup(srcLanguageID)(input._1._3._1),
            lengths = input._1._3._2)
          val mappedTgtSequences = Sequences(
            sequences = parameterManager.stringToIndexLookup(tgtLanguageID)(input._2._1),
            lengths = input._2._2)

          // Encode the source sentences.
          val encoderOutput = tf.variableScope("Encoder") {
            encoder(mappedSrcSequences)(context = ModelConstructionContext(
              stage = Encoding,
              mode = mode,
              srcLanguageID = srcLanguageID,
              tgtLanguageID = tgtLanguageID,
              tgtSequences = Some(mappedTgtSequences)))
          }

          // Decode to obtain the target sentences.
          val decoderOutput = tf.variableScope("Decoder") {
            decoder.applyTrain(encoderOutput)(context = ModelConstructionContext(
              stage = Decoding,
              mode = mode,
              srcLanguageID = srcLanguageID,
              tgtLanguageID = tgtLanguageID,
              tgtSequences = Some(mappedTgtSequences)))
          }

          (tgtLanguageID, (decoderOutput.sequences, decoderOutput.lengths))
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
        trainingConfig.curriculum.initialize()
        tf.createWith(device = deviceManager.nextDevice(env, moveToNext = false)) {
          parameterManager.initialize(languages, trainingConfig)

          val srcLanguageID = input._1
          val tgtLanguageID = input._2

          // Map tokens to token indices in the vocabulary.
          val mappedSrcSequences = Sequences(
            sequences = parameterManager.stringToIndexLookup(srcLanguageID)(input._3._1),
            lengths = input._3._2)

          // Perform inference based on the current pivoting strategy for multi-lingual translations.
          inferenceConfig.pivot match {
            case NoPivot =>
              // Encode the source sentences.
              val encoderOutput = tf.variableScope("Encoder") {
                encoder(mappedSrcSequences)(context = ModelConstructionContext(
                  stage = Encoding,
                  mode = mode,
                  srcLanguageID = srcLanguageID,
                  tgtLanguageID = tgtLanguageID,
                  tgtSequences = None))
              }

              // Decode to obtain the target sentences.
              val decoderOutput = tf.variableScope("Decoder") {
                decoder.applyInfer(encoderOutput)(context = ModelConstructionContext(
                  stage = Decoding,
                  mode = mode,
                  srcLanguageID = srcLanguageID,
                  tgtLanguageID = tgtLanguageID,
                  tgtSequences = None))
              }

              // Map from token indices in the vocabulary, back to actual tokens and return the result.
              val decodedSequences = decoderOutput.copy(
                sequences = parameterManager.indexToStringLookup(tgtLanguageID)(decoderOutput.sequences))
              (tgtLanguageID, (decodedSequences.sequences, decodedSequences.lengths))
            case _ =>
              inferenceConfig.pivot.initialize(languages, parameterManager)

              // Construct a pivoting sequence over languages and loop over it using a while loop.
              val pivotingSequence = inferenceConfig.pivot.pivotingSequence(srcLanguageID, tgtLanguageID)

              type LoopVariables = (Output[Int], Output[Int], Sentences[Int])

              def predicateFn(loopVariables: LoopVariables): Output[Boolean] = {
                tf.less(loopVariables._1, tf.shape(pivotingSequence).slice(0).toInt)
              }

              def bodyFn(loopVariables: LoopVariables): LoopVariables = {
                val index = loopVariables._1

                // Obtain the source and target language in this step of the pivoting chain.
                val currentSrcLanguage = loopVariables._2
                val currentTgtLanguage = tf.gather(pivotingSequence, index)

                // Encode the source sentences in this step of the pivoting chain.
                val encoderOutput = tf.variableScope("Encoder") {
                  val srcSequences = Sequences(
                    sequences = loopVariables._3._1,
                    lengths = loopVariables._3._2)
                  encoder(srcSequences)(context = ModelConstructionContext(
                    stage = Encoding,
                    mode = mode,
                    srcLanguageID = currentSrcLanguage,
                    tgtLanguageID = currentTgtLanguage,
                    tgtSequences = None))
                }

                // Decode to obtain the target sentences in this step of the pivoting chain.
                val decoderOutput = tf.variableScope("Decoder") {
                  decoder.applyInfer(encoderOutput)(context = ModelConstructionContext(
                    stage = Decoding,
                    mode = mode,
                    srcLanguageID = currentSrcLanguage,
                    tgtLanguageID = currentTgtLanguage,
                    tgtSequences = None))
                }

                (index + 1, currentTgtLanguage, (decoderOutput.sequences, decoderOutput.lengths))
              }

              val results = tf.whileLoop(
                predicateFn, bodyFn,
                loopVariables = (Output.constant[Int](0), srcLanguageID, (mappedSrcSequences.sequences, mappedSrcSequences.lengths)),
                shapeInvariants = Some((Shape(), Shape(), (Shape(-1, -1), Shape(-1)))))

              // Map from token indices in the vocabulary, back to actual tokens and return the result.
              val decodedSequences = parameterManager.indexToStringLookup(tgtLanguageID)(results._3._1)
              (tgtLanguageID, (decodedSequences, /* target sentence lengths */ results._3._2))
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
        val globalStep = Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false)
        tf.createWith(
          nameScope = "Loss",
          device = deviceManager.nextDevice(env, moveToNext = false),
          controlDependencies = Set(trainingConfig.curriculum.updateState(globalStep.value))
        ) {
          parameterManager.initialize(languages, trainingConfig)

          val tgtLanguage = input._1._1

          var tgtSequences = tf.nameScope("TrainInputsToTokenIDs") {
            // Map the reference tokens to token indices in the vocabulary.
            parameterManager.stringToIndexLookup(tgtLanguage)(input._2._2._1)
          }

          // TODO: Handle this shift more efficiently.
          // Shift the target sequence one step backward so the decoder is evaluated based using the correct previous
          // token used as input, rather than the previous predicted token.
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

  protected def loss(
      predictedSequences: Output[Float],
      tgtSequences: Output[Int],
      tgtSequenceLengths: Output[Int]
  ): Output[Float] = {
    val (lossSum, _) = Common.paddedCrossEntropy(
      logits = predictedSequences,
      labels = tgtSequences,
      labelLengths = tgtSequenceLengths,
      labelSmoothing = trainingConfig.labelSmoothing)
    lossSum / tf.size(tgtSequenceLengths).toFloat
  }
}
