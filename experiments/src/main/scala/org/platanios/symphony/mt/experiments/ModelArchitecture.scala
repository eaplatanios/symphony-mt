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
import org.platanios.symphony.mt.data.{DataConfig, FileParallelDataset}
import org.platanios.symphony.mt.evaluation.MTMetric
import org.platanios.symphony.mt.models.helpers.decoders.GoogleLengthPenalty
import org.platanios.symphony.mt.models.parameters.{PairwiseManager, ParameterManager}
import org.platanios.symphony.mt.models.pivoting.{NoPivot, Pivot, SinglePivot}
import org.platanios.symphony.mt.models.{Model, ModelConfig}
import org.platanios.symphony.mt.models.rnn._
import org.platanios.symphony.mt.models.rnn.attention.{BahdanauRNNAttention, LuongRNNAttention}
import org.platanios.symphony.mt.models.transformer.{TransformerEncoder, TransformerDecoder}
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsDecimal, IsReal, TF}
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape, Zero}

/**
  * @author Emmanouil Antonios Platanios
  */
sealed trait ModelArchitecture {
  val name: String

  def model[T: TF : IsHalfOrFloatOrDouble](
      name: String,
      languages: Seq[(Language, Vocabulary)],
      dataConfig: DataConfig,
      env: Environment,
      parameterManager: ParameterManager,
      trainIdentityTranslations: Boolean,
      languagePairs: Set[(Language, Language)],
      evalLanguagePairs: Set[(Language, Language)],
      cellString: String,
      numUnits: Int,
      residual: Boolean,
      dropout: Option[Float],
      attention: Boolean,
      labelSmoothing: Float,
      summarySteps: Int,
      checkpointSteps: Int,
      beamWidth: Int,
      lengthPenaltyWeight: Float,
      maxDecodingLengthFactor: Float,
      optConfig: ModelConfig.OptConfig,
      logConfig: ModelConfig.LogConfig,
      evalDatasets: Seq[(String, FileParallelDataset, Float)],
      evalMetrics: Seq[MTMetric]
  ): Model[_] = {
    val cell = cellFromString[T](cellString)
    createModel[T, cell.StateType, cell.StateShapeType](
      name, languages, dataConfig, env, parameterManager, trainIdentityTranslations, languagePairs, evalLanguagePairs,
      cell.asInstanceOf[Cell[cell.DataType, cell.StateType, cell.StateShapeType]], numUnits, residual, dropout,
      attention, labelSmoothing, summarySteps, checkpointSteps, beamWidth, lengthPenaltyWeight, maxDecodingLengthFactor,
      optConfig, logConfig, evalDatasets, evalMetrics
    )(
      TF[T], IsHalfOrFloatOrDouble[T],
      cell.evOutputStructureState.asInstanceOf[OutputStructure[cell.StateType]],
      cell.evOutputToShapeState.asInstanceOf[OutputToShape.Aux[cell.StateType, cell.StateShapeType]],
      cell.evZeroState.asInstanceOf[Zero.Aux[cell.StateType, cell.StateShapeType]])
  }

  protected def createModel[T: TF : IsHalfOrFloatOrDouble, State: OutputStructure, StateShape](
      name: String,
      languages: Seq[(Language, Vocabulary)],
      dataConfig: DataConfig,
      env: Environment,
      parameterManager: ParameterManager,
      trainIdentityTranslations: Boolean,
      languagePairs: Set[(Language, Language)],
      evalLanguagePairs: Set[(Language, Language)],
      cell: Cell[T, State, StateShape],
      numUnits: Int,
      residual: Boolean,
      dropout: Option[Float],
      attention: Boolean,
      labelSmoothing: Float,
      summarySteps: Int,
      checkpointSteps: Int,
      beamWidth: Int,
      lengthPenaltyWeight: Float,
      maxDecodingLengthFactor: Float,
      optConfig: ModelConfig.OptConfig,
      logConfig: ModelConfig.LogConfig,
      evalDatasets: Seq[(String, FileParallelDataset, Float)],
      evalMetrics: Seq[MTMetric]
  )(implicit
      evOutputToShapeState: OutputToShape.Aux[State, StateShape],
      evZeroState: Zero.Aux[State, StateShape]
  ): Model[_]

  protected def cellFromString[T: TF : IsReal](cellString: String): Cell[T, _, _] = {
    val parts = cellString.split(":")
    val activation: Output[T] => Output[T] = {
      if (parts.length < 2) {
        tf.tanh(_)
      } else {
        parts(1) match {
          case "sigmoid" => tf.sigmoid(_)
          case "tanh" => tf.tanh(_)
          case "relu" => tf.relu(_)
          case "relu6" => tf.relu6(_)
          case "elu" => tf.elu(_)
          case "selu" => tf.selu(_)
          case _ => throw new IllegalArgumentException(
            s"'${parts(1)}' does not represent a valid activation function.")
        }
      }
    }
    parts(0) match {
      case "gru" => GRU(activation)
      case "lstm" if parts.length == 2 => BasicLSTM(activation = activation)
      case "lstm" if parts.length == 3 => BasicLSTM(activation = activation, forgetBias = parts(2).toFloat)
      case _ => throw new IllegalArgumentException(s"'$cellString' does not represent a valid RNN cell type.")
    }
  }

  override def toString: String
}

object ModelArchitecture {
  implicit val modelArchitectureRead: scopt.Read[ModelArchitecture] = {
    scopt.Read.reads(value => {
      val parts = value.split(":")
      parts(0) match {
        case "rnn" if parts.length == 3 => RNN(parts(1).toInt, parts(2).toInt)
        case "bi_rnn" if parts.length == 3 => BiRNN(parts(1).toInt, parts(2).toInt)
        case "gnmt" if parts.length == 4 => GNMT(parts(1).toInt, parts(2).toInt, parts(3).toInt)
        case "transformer" if parts.length == 3 => Transformer(parts(1).toInt, parts(2).toInt)
        case _ => throw new IllegalArgumentException(s"'$value' does not represent a valid model architecture.")
      }
    })
  }

  private[experiments] def pivot(
      parameterManager: ParameterManager,
      languagePairs: Set[(Language, Language)]
  ): Pivot = {
    parameterManager match {
      case _: PairwiseManager => SinglePivot(Language.English, languagePairs)
      case _ => NoPivot
    }
  }
}

case class RNN(
    numEncoderLayers: Int = 2,
    numDecoderLayers: Int = 2
) extends ModelArchitecture {
  override val name: String = "rnn"

  override protected def createModel[T: TF : IsHalfOrFloatOrDouble, State: OutputStructure, StateShape](
      name: String,
      languages: Seq[(Language, Vocabulary)],
      dataConfig: DataConfig,
      env: Environment,
      parameterManager: ParameterManager,
      trainIdentityTranslations: Boolean,
      languagePairs: Set[(Language, Language)],
      evalLanguagePairs: Set[(Language, Language)],
      cell: Cell[T, State, StateShape],
      numUnits: Int,
      residual: Boolean,
      dropout: Option[Float],
      attention: Boolean,
      labelSmoothing: Float,
      summarySteps: Int,
      checkpointSteps: Int,
      beamWidth: Int,
      lengthPenaltyWeight: Float,
      maxDecodingLengthFactor: Float,
      optConfig: ModelConfig.OptConfig,
      logConfig: ModelConfig.LogConfig,
      evalDatasets: Seq[(String, FileParallelDataset, Float)],
      evalMetrics: Seq[MTMetric]
  )(implicit
      evOutputToShapeState: OutputToShape.Aux[State, StateShape],
      evZeroState: Zero.Aux[State, StateShape]
  ): Model[_] = {
    new Model(
      name = name,
      languages = languages,
      encoder = new UnidirectionalRNNEncoder(
        cell = cell,
        numUnits = numUnits,
        numLayers = numEncoderLayers,
        residual = residual,
        dropout = dropout),
      decoder = if (attention) {
        new UnidirectionalRNNDecoderWithAttention(
          cell = cell,
          numUnits = numUnits,
          numLayers = numDecoderLayers,
          residual = residual,
          dropout = dropout,
          attention = new LuongRNNAttention(
            scaled = true,
            probabilityFn = (o: Output[T]) => tf.softmax(o)),
          outputAttention = attention)
      } else {
        new UnidirectionalRNNDecoder(
          cell = cell,
          numUnits = numUnits,
          numLayers = numDecoderLayers,
          residual = residual,
          dropout = dropout)
      },
      env = env,
      parameterManager = parameterManager,
      dataConfig = dataConfig,
      modelConfig = ModelConfig(
        pivot = ModelArchitecture.pivot(parameterManager, languagePairs),
        optConfig = optConfig,
        logConfig = logConfig,
        beamWidth = beamWidth,
        lengthPenalty = GoogleLengthPenalty(lengthPenaltyWeight),
        maxDecodingLengthFactor = maxDecodingLengthFactor,
        labelSmoothing = labelSmoothing,
        summarySteps = summarySteps,
        checkpointSteps = checkpointSteps,
        trainIdentityTranslations = trainIdentityTranslations,
        languagePairs = languagePairs,
        evalLanguagePairs = evalLanguagePairs)
    )(
      evalDatasets = evalDatasets,
      evalMetrics = evalMetrics)
  }

  override def toString: String = s"rnn:$numEncoderLayers:$numDecoderLayers"
}

case class BiRNN(
    numEncoderLayers: Int = 2,
    numDecoderLayers: Int = 2
) extends ModelArchitecture {
  override val name: String = "bi_rnn"

  override protected def createModel[T: TF : IsHalfOrFloatOrDouble, State: OutputStructure, StateShape](
      name: String,
      languages: Seq[(Language, Vocabulary)],
      dataConfig: DataConfig,
      env: Environment,
      parameterManager: ParameterManager,
      trainIdentityTranslations: Boolean,
      languagePairs: Set[(Language, Language)],
      evalLanguagePairs: Set[(Language, Language)],
      cell: Cell[T, State, StateShape],
      numUnits: Int,
      residual: Boolean,
      dropout: Option[Float],
      attention: Boolean,
      labelSmoothing: Float,
      summarySteps: Int,
      checkpointSteps: Int,
      beamWidth: Int,
      lengthPenaltyWeight: Float,
      maxDecodingLengthFactor: Float,
      optConfig: ModelConfig.OptConfig,
      logConfig: ModelConfig.LogConfig,
      evalDatasets: Seq[(String, FileParallelDataset, Float)],
      evalMetrics: Seq[MTMetric]
  )(implicit
      evOutputToShapeState: OutputToShape.Aux[State, StateShape],
      evZeroState: Zero.Aux[State, StateShape]
  ): Model[_] = {
    new Model(
      name = name,
      languages = languages,
      encoder = new BidirectionalRNNEncoder(
        cell = cell,
        numUnits = numUnits,
        numLayers = numEncoderLayers,
        residual = residual,
        dropout = dropout),
      decoder = if (attention) {
        new UnidirectionalRNNDecoderWithAttention(
          cell = cell,
          numUnits = numUnits,
          numLayers = numDecoderLayers,
          residual = residual,
          dropout = dropout,
          attention = new LuongRNNAttention(
            scaled = true,
            probabilityFn = (o: Output[T]) => tf.softmax(o)),
          outputAttention = attention)
      } else {
        new UnidirectionalRNNDecoder(
          cell = cell,
          numUnits = numUnits,
          numLayers = numDecoderLayers,
          residual = residual,
          dropout = dropout)
      },
      env = env,
      parameterManager = parameterManager,
      dataConfig = dataConfig,
      modelConfig = ModelConfig(
        pivot = ModelArchitecture.pivot(parameterManager, languagePairs),
        optConfig = optConfig,
        logConfig = logConfig,
        beamWidth = beamWidth,
        lengthPenalty = GoogleLengthPenalty(lengthPenaltyWeight),
        maxDecodingLengthFactor = maxDecodingLengthFactor,
        labelSmoothing = labelSmoothing,
        summarySteps = summarySteps,
        checkpointSteps = checkpointSteps,
        trainIdentityTranslations = trainIdentityTranslations,
        languagePairs = languagePairs,
        evalLanguagePairs = evalLanguagePairs)
    )(
      evalDatasets = evalDatasets,
      evalMetrics = evalMetrics)
  }

  override def toString: String = s"bi_rnn:$numEncoderLayers:$numDecoderLayers"
}

case class GNMT(
    numBiLayers: Int = 1,
    numUniLayers: Int = 3,
    numUniResLayers: Int = 2
) extends ModelArchitecture {
  override val name: String = "gnmt"

  override protected def createModel[T: TF : IsHalfOrFloatOrDouble, State: OutputStructure, StateShape](
      name: String,
      languages: Seq[(Language, Vocabulary)],
      dataConfig: DataConfig,
      env: Environment,
      parameterManager: ParameterManager,
      trainIdentityTranslations: Boolean,
      languagePairs: Set[(Language, Language)],
      evalLanguagePairs: Set[(Language, Language)],
      cell: Cell[T, State, StateShape],
      numUnits: Int,
      residual: Boolean,
      dropout: Option[Float],
      attention: Boolean,
      labelSmoothing: Float,
      summarySteps: Int,
      checkpointSteps: Int,
      beamWidth: Int,
      lengthPenaltyWeight: Float,
      maxDecodingLengthFactor: Float,
      optConfig: ModelConfig.OptConfig,
      logConfig: ModelConfig.LogConfig,
      evalDatasets: Seq[(String, FileParallelDataset, Float)],
      evalMetrics: Seq[MTMetric]
  )(implicit
      evOutputToShapeState: OutputToShape.Aux[State, StateShape],
      evZeroState: Zero.Aux[State, StateShape]
  ): Model[_] = {
    new Model(
      name = name,
      languages = languages,
      encoder = new GNMTEncoder(
        cell = cell,
        numUnits = numUnits,
        numBiLayers = numBiLayers,
        numUniLayers = numUniLayers,
        numUniResLayers = numUniResLayers,
        dropout = dropout),
      decoder = new GNMTDecoder(
        cell = cell,
        numUnits = numUnits,
        numLayers = numBiLayers + numUniLayers,
        numResLayers = numUniResLayers,
        attention = new BahdanauRNNAttention(
          normalized = true,
          probabilityFn = (o: Output[T]) => tf.softmax(o)),
        dropout = dropout,
        useNewAttention = attention),
      env = env,
      parameterManager = parameterManager,
      dataConfig = dataConfig,
      modelConfig = ModelConfig(
        pivot = ModelArchitecture.pivot(parameterManager, languagePairs),
        optConfig = optConfig,
        logConfig = logConfig,
        beamWidth = beamWidth,
        lengthPenalty = GoogleLengthPenalty(lengthPenaltyWeight),
        maxDecodingLengthFactor = maxDecodingLengthFactor,
        labelSmoothing = labelSmoothing,
        summarySteps = summarySteps,
        checkpointSteps = checkpointSteps,
        trainIdentityTranslations = trainIdentityTranslations,
        languagePairs = languagePairs,
        evalLanguagePairs = evalLanguagePairs)
    )(
      evalDatasets = evalDatasets,
      evalMetrics = evalMetrics)
  }

  override def toString: String = s"gnmt:$numBiLayers:$numUniLayers:$numUniResLayers"
}

case class Transformer(
    encoderNumLayers: Int = 6,
    decoderNumLayers: Int = 6
) extends ModelArchitecture {
  override val name: String = "transformer"

  override protected def createModel[T: TF : IsHalfOrFloatOrDouble, State: OutputStructure, StateShape](
      name: String,
      languages: Seq[(Language, Vocabulary)],
      dataConfig: DataConfig,
      env: Environment,
      parameterManager: ParameterManager,
      trainIdentityTranslations: Boolean,
      languagePairs: Set[(Language, Language)],
      evalLanguagePairs: Set[(Language, Language)],
      cell: Cell[T, State, StateShape],
      numUnits: Int,
      residual: Boolean,
      dropout: Option[Float],
      attention: Boolean,
      labelSmoothing: Float,
      summarySteps: Int,
      checkpointSteps: Int,
      beamWidth: Int,
      lengthPenaltyWeight: Float,
      maxDecodingLengthFactor: Float,
      optConfig: ModelConfig.OptConfig,
      logConfig: ModelConfig.LogConfig,
      evalDatasets: Seq[(String, FileParallelDataset, Float)],
      evalMetrics: Seq[MTMetric]
  )(implicit
      evOutputToShapeState: OutputToShape.Aux[State, StateShape],
      evZeroState: Zero.Aux[State, StateShape]
  ): Model[_] = {
    new Model(
      name = name,
      languages = languages,
      encoder = new TransformerEncoder[T](
        numUnits = numUnits,
        numLayers = encoderNumLayers),
      decoder = new TransformerDecoder[T](
        numLayers = decoderNumLayers),
      env = env,
      parameterManager = parameterManager,
      dataConfig = dataConfig,
      modelConfig = ModelConfig(
        pivot = ModelArchitecture.pivot(parameterManager, languagePairs),
        optConfig = optConfig,
        logConfig = logConfig,
        beamWidth = beamWidth,
        lengthPenalty = GoogleLengthPenalty(lengthPenaltyWeight),
        maxDecodingLengthFactor = maxDecodingLengthFactor,
        labelSmoothing = labelSmoothing,
        summarySteps = summarySteps,
        checkpointSteps = checkpointSteps,
        trainIdentityTranslations = trainIdentityTranslations,
        languagePairs = languagePairs,
        evalLanguagePairs = evalLanguagePairs)
    )(
      evalDatasets = evalDatasets,
      evalMetrics = evalMetrics)
  }

  override def toString: String = s"transformer:$encoderNumLayers:$decoderNumLayers"
}
