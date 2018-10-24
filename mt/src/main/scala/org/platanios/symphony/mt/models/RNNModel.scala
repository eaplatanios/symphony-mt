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
import org.platanios.symphony.mt.evaluation._
import org.platanios.symphony.mt.models.parameters.ParameterManager
import org.platanios.symphony.mt.models.pivoting.{NoPivot, Pivot}
import org.platanios.symphony.mt.models.rnn.{Cell, RNNDecoder, RNNEncoder}
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple

/**
  * @author Emmanouil Antonios Platanios
  */
class RNNModel[T: TF : IsNotQuantized, State : OutputStructure](
    override val name: String = "RNNModel",
    override val languages: Seq[(Language, Vocabulary)],
    override val dataConfig: DataConfig,
    override val config: RNNModel.Config[T, State],
    override val optConfig: Model.OptConfig,
    override val logConfig : Model.LogConfig  = Model.LogConfig()
)(
    override val evalDatasets: Seq[(String, FileParallelDataset, Float)] = Seq.empty,
    override val evalMetrics: Seq[MTMetric] = Seq(
      BLEU()(languages),
      Meteor()(languages),
      TER()(languages),
      SentenceLength(forHypothesis = true, name = "HypLen"),
      SentenceLength(forHypothesis = false, name = "RefLen"),
      SentenceCount(name = "#Sentences"))
) extends Model[(Tuple[Output[T], Seq[State]], Output[Int], Output[Int])](
  name, languages, dataConfig, config, optConfig, logConfig
)(
  evalDatasets, evalMetrics
) {
  // TODO: Make this use the parameters manager.

  override protected def encoder(input: SentencesWithLanguagePair[Int])(implicit
      mode: Mode,
      context: Output[Int]
  ): (Tuple[Output[T], Seq[State]], Output[Int], Output[Int]) = {
    implicit val stage: Stage = Encoding

    val (srcSentences, srcSentenceLengths) = input._3

    val maxDecodingLength = {
      if (!mode.isTraining && dataConfig.tgtMaxLength != -1)
        tf.constant(dataConfig.tgtMaxLength)
      else
        tf.round(tf.max(tf.max(srcSentenceLengths)).toFloat * config.decoderMaxLengthFactor).toInt
    }
    (config.encoder.create(config, srcSentences, srcSentenceLengths), srcSentenceLengths, maxDecodingLength)
  }

  override protected def decoder[O: TF](
      decodingMode: Model.DecodingMode[O],
      encoderInput: (LanguageID, LanguageID, (Output[Int], SentenceLengths)),
      input: Option[(Output[Int], SentenceLengths)],
      state: Option[(Tuple[Output[T], Seq[State]], Output[Int], Output[Int])]
  )(implicit
      mode: Mode,
      context: Output[Int]
  ): (Output[O], SentenceLengths) = {
    implicit val stage: Stage = Decoding

    // TODO: What if the state is `None`?
    input match {
      case Some(inputSequences) =>
        // TODO: Handle this shift more efficiently.
        // Shift the target sequence one step forward so the decoder learns to output the next word.
        val tgtBosId = config.parameterManager
            .stringToIndexLookup(encoderInput._2)(tf.constant(dataConfig.beginOfSequenceToken))
        val batchSize = tf.shape(inputSequences._1).slice(0).toInt
        val tgtSequence = tf.concatenate(Seq(
          tf.fill(INT32, tf.stack[Int](Seq(batchSize, 1)))(tgtBosId),
          inputSequences._1), axis = 1)
        val tgtSequenceLength = inputSequences._2 + 1
        val output = config.decoder.create(
          decodingMode, config, state.get, dataConfig.beginOfSequenceToken,
          dataConfig.endOfSequenceToken, tgtSequence, tgtSequenceLength)
        (output.sequences, output.lengths)
      case None =>
        val output = config.decoder.create(
          decodingMode, config, state.get, dataConfig.beginOfSequenceToken,
          dataConfig.endOfSequenceToken)
        // Make sure the outputs are of shape [batchSize, time] or [beamWidth, batchSize, time]
        // when using beam search.
        val outputSequence = {
          if (config.timeMajor)
            output.sequences.transpose()
          else if (output.sequences.rank == 3)
            output.sequences.transpose(Tensor(2, 0, 1))
          else
            output.sequences
        }
        (outputSequence(---, 0 :: -1), output.lengths - 1)
    }
  }
}

object RNNModel {
  def apply[T: TF : IsNotQuantized, State: OutputStructure](
      name: String = "RNNModel",
      languages: Seq[(Language, Vocabulary)],
      dataConfig: DataConfig,
      config: RNNModel.Config[T, State],
      optConfig: Model.OptConfig,
      logConfig: Model.LogConfig
  )(
      evalDatasets: Seq[(String, FileParallelDataset, Float)] = Seq.empty,
      evalMetrics: Seq[MTMetric] = Seq(
        BLEU()(languages),
        Meteor()(languages),
        TER()(languages),
        SentenceLength(forHypothesis = true, name = "HypLen"),
        SentenceLength(forHypothesis = false, name = "RefLen"),
        SentenceCount(name = "#Sentences"))
  ): RNNModel[T, State] = {
    new RNNModel[T, State](
      name, languages, dataConfig, config, optConfig, logConfig)(evalDatasets, evalMetrics)
  }

  class Config[T, S] protected(
      override val env: Environment,
      override val parameterManager: ParameterManager,
      override val deviceManager: DeviceManager,
      override val pivot: Pivot,
      override val labelSmoothing: Float,
      // Model
      val encoder: RNNEncoder[T, S],
      val decoder: RNNDecoder[T, S],
      override val timeMajor: Boolean,
      override val summarySteps: Int,
      override val checkpointSteps: Int,
      override val trainIdentityTranslations: Boolean,
      override val languagePairs: Set[(Language, Language)],
      override val evalLanguagePairs: Set[(Language, Language)],
      // Inference
      val beamWidth: Int,
      val lengthPenaltyWeight: Float,
      val decoderMaxLengthFactor: Float
  ) extends Model.Config(
    env, parameterManager, deviceManager, pivot, labelSmoothing, timeMajor, summarySteps, checkpointSteps,
    trainIdentityTranslations, languagePairs, evalLanguagePairs)

  object Config {
    def apply[T, S](
        env: Environment,
        parameterManager: ParameterManager,
        // Model
        encoder: RNNEncoder[T, S],
        decoder: RNNDecoder[T, S],
        deviceManager: DeviceManager = RoundRobinDeviceManager,
        pivot: Pivot = NoPivot,
        timeMajor: Boolean = false,
        labelSmoothing: Float = 0.1f,
        summarySteps: Int = 100,
        checkpointSteps: Int = 1000,
        trainBackTranslation: Boolean = false,
        languagePairs: Set[(Language, Language)] = Set.empty,
        evalLanguagePairs: Set[(Language, Language)] = Set.empty,
        // Inference
        beamWidth: Int = 10,
        lengthPenaltyWeight: Float = 0.0f,
        decoderMaxLengthFactor: Float = 2.0f
    ): Config[T, S] = {
      new Config[T, S](
        env, parameterManager, deviceManager, pivot, labelSmoothing, encoder, decoder, timeMajor, summarySteps,
        checkpointSteps, trainBackTranslation, languagePairs, evalLanguagePairs, beamWidth, lengthPenaltyWeight,
        decoderMaxLengthFactor)
    }
  }

  private[models] def cell[T: TF : IsNotQuantized, State: OutputStructure, StateShape](
      cell: Cell[T, State, StateShape],
      numInputs: Int,
      numUnits: Int,
      dropout: Option[Float] = None,
      residualFn: Option[(Output[T], Output[T]) => Output[T]] = None,
      device: String = "",
      seed: Option[Int] = None,
      name: String
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      stage: Stage,
      context: Output[Int],
      evOutputToShape: OutputToShape.Aux[State, StateShape]
  ): tf.RNNCell[Output[T], State, Shape, StateShape] = {
    tf.variableScope(name) {
      tf.createWith(device = device) {
        // Create the main RNN cell.
        var createdCell = cell.create(name, numInputs, numUnits)

        // Apply dropout.
        createdCell = dropout.map(p => {
          if (!mode.isTraining) {
            createdCell
          } else {
            tf.DropoutWrapper(createdCell, 1.0f - p, seed = seed, name = "Dropout")
          }
        }).getOrElse(createdCell)

        // Add residual connections.
        createdCell = residualFn.map(tf.ResidualWrapper(createdCell, _)).getOrElse(createdCell)

        createdCell
      }
    }
  }

  private[models] def cells[T: TF : IsNotQuantized, State: OutputStructure, StateShape](
      cell: Cell[T, State, StateShape],
      numInputs: Int,
      numUnits: Int,
      numLayers: Int,
      numResidualLayers: Int,
      dropout: Option[Float] = None,
      residualFn: Option[(Output[T], Output[T]) => Output[T]] = None,
      seed: Option[Int] = None,
      name: String
  )(implicit
      mode: Mode,
      env: Environment,
      parameterManager: ParameterManager,
      deviceManager: DeviceManager,
      stage: Stage,
      context: Output[Int],
      evOutputToShape: OutputToShape.Aux[State, StateShape]
  ): Seq[tf.RNNCell[Output[T], State, Shape, StateShape]] = {
    tf.variableScope(name) {
      (0 until numLayers).foldLeft(Seq.empty[tf.RNNCell[Output[T], State, Shape, StateShape]])((cells, i) => {
        val cellNumInputs = if (i == 0) numInputs else cells(i - 1).outputShape(-1)
        cells :+ this.cell[T, State, StateShape](
          cell, cellNumInputs, numUnits, dropout,
          if (i >= numLayers - numResidualLayers) residualFn else None,
          deviceManager.nextDevice(env), seed, s"Cell$i")
      })
    }
  }

  private[models] def stackedCell[T: TF : IsNotQuantized, State: OutputStructure, StateShape](
      cell: Cell[T, State, StateShape],
      numInputs: Int,
      numUnits: Int,
      numLayers: Int,
      numResidualLayers: Int,
      dropout: Option[Float] = None,
      residualFn: Option[(Output[T], Output[T]) => Output[T]] = None,
      seed: Option[Int] = None,
      name: String
  )(implicit
      mode: Mode,
      env: Environment,
      parameterManager: ParameterManager,
      deviceManager: DeviceManager,
      stage: Stage,
      context: Output[Int],
      evOutputToShape: OutputToShape.Aux[State, StateShape]
  ): tf.RNNCell[Output[T], Seq[State], Shape, Seq[StateShape]] = {
    tf.StackedCell(cells[T, State, StateShape](
      cell, numInputs, numUnits, numLayers, numResidualLayers, dropout,
      residualFn, seed, name), name)
  }
}
