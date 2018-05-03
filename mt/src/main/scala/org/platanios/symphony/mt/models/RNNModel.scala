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
import org.platanios.symphony.mt.models.rnn.{Cell, RNNDecoder, RNNEncoder}
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple

/**
  * @author Emmanouil Antonios Platanios
  */
class RNNModel[S, SS](
    override val name: String = "RNNModel",
    override val languages: Seq[(Language, Vocabulary)],
    override val dataConfig: DataConfig,
    override val config: RNNModel.Config[S, SS],
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
)(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) extends Model[(Tuple[Output, Seq[S]], Output, Output)](
  name, languages, dataConfig, config, optConfig, logConfig
)(
  evalDatasets, evalMetrics
) {
  // TODO: Make this use the parameters manager.

  override protected def encoder(
      input: TFBatchWithLanguages
  )(implicit mode: Mode): (Tuple[Output, Seq[S]], Output, Output) = {
    implicit val stage: Stage = Encoding

    val maxDecodingLength = {
      if (!mode.isTraining && dataConfig.tgtMaxLength != -1)
        tf.constant(dataConfig.tgtMaxLength)
      else
        tf.round(tf.max(tf.max(input._4)) * config.decoderMaxLengthFactor).cast(INT32)
    }
    (config.encoder.create(config, input._1, input._2, input._3, input._4), input._4, maxDecodingLength)
  }

  override protected def decoder(
      encoderInput: TFBatchWithLanguages,
      input: Option[TFBatch],
      state: Option[(Tuple[Output, Seq[S]], Output, Output)]
  )(implicit mode: Mode): TFBatch = {
    implicit val stage: Stage = Decoding

    // TODO: What if the state is `None`?
    input match {
      case Some(inputSequences) =>
        // TODO: Handle this shift more efficiently.
        // Shift the target sequence one step forward so the decoder learns to output the next word.
        val tgtBosId = config.parameterManager
            .stringToIndexLookup(encoderInput._2)(tf.constant(dataConfig.beginOfSequenceToken)).cast(INT32)
        val tgtSequence = tf.concatenate(Seq(
          tf.fill(INT32, tf.stack(Seq(tf.shape(inputSequences._1)(0), 1)))(tgtBosId),
          inputSequences._1), axis = 1)
        val tgtSequenceLength = inputSequences._2 + 1
        val output = config.decoder.create(
          config, encoderInput._1, encoderInput._2, state.get, dataConfig.beginOfSequenceToken,
          dataConfig.endOfSequenceToken, tgtSequence, tgtSequenceLength)
        (output.sequences, output.sequenceLengths)
      case None =>
        val output = config.decoder.create(
          config, encoderInput._1, encoderInput._2, state.get, dataConfig.beginOfSequenceToken,
          dataConfig.endOfSequenceToken, null, null)
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
        (outputSequence(---, 0 :: -1), output.sequenceLengths - 1)
    }
  }
}

object RNNModel {
  def apply[S, SS](
      name: String = "RNNModel",
      languages: Seq[(Language, Vocabulary)],
      dataConfig: DataConfig,
      config: RNNModel.Config[S, SS],
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
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): RNNModel[S, SS] = {
    new RNNModel[S, SS](
      name, languages, dataConfig, config, optConfig, logConfig)(evalDatasets, evalMetrics)(evS, evSDropout)
  }

  class Config[S, SS] protected(
      override val env: Environment,
      override val parameterManager: ParameterManager,
      override val deviceManager: DeviceManager,
      override val labelSmoothing: Float,
      // Model
      val encoder: RNNEncoder[S, SS],
      val decoder: RNNDecoder[S, SS],
      override val timeMajor: Boolean,
      override val summarySteps: Int,
      override val checkpointSteps: Int,
      override val trainBackTranslation: Boolean,
      override val languagePairs: Set[(Language, Language)],
      override val evalLanguagePairs: Set[(Language, Language)],
      // Inference
      val beamWidth: Int,
      val lengthPenaltyWeight: Float,
      val decoderMaxLengthFactor: Float
  ) extends Model.Config(
    env, parameterManager, deviceManager, labelSmoothing, timeMajor, summarySteps, checkpointSteps,
    trainBackTranslation, languagePairs, evalLanguagePairs)

  object Config {
    def apply[S, SS](
        env: Environment,
        parameterManager: ParameterManager,
        // Model
        encoder: RNNEncoder[S, SS],
        decoder: RNNDecoder[S, SS],
        deviceManager: DeviceManager = RoundRobinDeviceManager,
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
    ): Config[S, SS] = {
      new Config[S, SS](
        env, parameterManager, deviceManager, labelSmoothing, encoder, decoder, timeMajor, summarySteps,
        checkpointSteps, trainBackTranslation, languagePairs, evalLanguagePairs, beamWidth, lengthPenaltyWeight,
        decoderMaxLengthFactor)
    }
  }

  private[models] def cell[S, SS](
      cellCreator: Cell[S, SS],
      numInputs: Int,
      numUnits: Int,
      dataType: DataType,
      dropout: Option[Float] = None,
      residualFn: Option[(Output, Output) => Output] = None,
      device: String = "",
      seed: Option[Int] = None,
      name: String
  )(mode: Mode, parameterManager: ParameterManager)(implicit
      stage: Stage,
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): tf.RNNCell[Output, Shape, S, SS] = tf.variableScope(name) {
    tf.createWith(device = device) {
      // Create the main RNN cell.
      var createdCell = cellCreator.create(name, numInputs, numUnits, dataType)(mode, parameterManager)

      // Apply dropout.
      createdCell = dropout.map(p => {
        if (!mode.isTraining)
          createdCell
        else
          tf.DropoutWrapper(createdCell, 1.0f - p, seed = seed, name = "Dropout")
      }).getOrElse(createdCell)

      // Add residual connections.
      createdCell = residualFn.map(tf.ResidualWrapper(createdCell, _)).getOrElse(createdCell)

      createdCell
    }
  }

  private[models] def cells[S, SS](
      cellCreator: Cell[S, SS],
      numInputs: Int,
      numUnits: Int,
      dataType: DataType,
      numLayers: Int,
      numResidualLayers: Int,
      dropout: Option[Float] = None,
      residualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output),
      seed: Option[Int] = None,
      name: String
  )(
      mode: Mode,
      env: Environment,
      parameterManager: ParameterManager,
      deviceManager: DeviceManager
  )(implicit
      stage: Stage,
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): Seq[tf.RNNCell[Output, Shape, S, SS]] = tf.variableScope(name) {
    (0 until numLayers).foldLeft(Seq.empty[tf.RNNCell[Output, Shape, S, SS]])((cells, i) => {
      val cellNumInputs = if (i == 0) numInputs else cells(i - 1).outputShape(-1)
      cells :+ cell(
        cellCreator, cellNumInputs, numUnits, dataType, dropout,
        if (i >= numLayers - numResidualLayers) residualFn else None,
        deviceManager.nextDevice(env), seed, s"Cell$i")(mode, parameterManager)
    })
  }

  private[models] def multiCell[S, SS](
      cellCreator: Cell[S, SS],
      numInputs: Int,
      numUnits: Int,
      dataType: DataType,
      numLayers: Int,
      numResidualLayers: Int,
      dropout: Option[Float] = None,
      residualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output),
      seed: Option[Int] = None,
      name: String
  )(
      mode: Mode,
      env: Environment,
      parameterManager: ParameterManager,
      deviceManager: DeviceManager
  )(implicit
      stage: Stage,
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): tf.RNNCell[Output, Shape, Seq[S], Seq[SS]] = {
    tf.MultiCell(cells(
      cellCreator, numInputs, numUnits, dataType, numLayers, numResidualLayers, dropout,
      residualFn, seed, name)(mode, env, parameterManager, deviceManager), name)
  }
}
