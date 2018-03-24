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
import org.platanios.symphony.mt.models.{Model, ParameterManager, RNNModel}
import org.platanios.symphony.mt.models.rnn._
import org.platanios.symphony.mt.models.rnn.attention.{BahdanauRNNAttention, LuongRNNAttention}
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable

/**
  * @author Emmanouil Antonios Platanios
  */
sealed trait ModelArchitecture {
  val name: String

  def model(
      name: String,
      languages: Seq[(Language, Vocabulary)],
      dataConfig: DataConfig,
      env: Environment,
      parameterManager: ParameterManager,
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
      decoderMaxLengthFactor: Float,
      optConfig: Model.OptConfig,
      logConfig: Model.LogConfig,
      evalDatasets: Seq[(String, FileParallelDataset)],
      evalMetrics: Seq[MTMetric]
  ): RNNModel[_, _] = {
    val cell = cellFromString(cellString)
    createModel[cell.StateType, cell.StateShapeType](
      name, languages, dataConfig, env, parameterManager, cell.asInstanceOf[Cell[cell.StateType, cell.StateShapeType]],
      numUnits, residual, dropout, attention, labelSmoothing, summarySteps, checkpointSteps, beamWidth,
      lengthPenaltyWeight, decoderMaxLengthFactor, optConfig, logConfig,
      evalDatasets, evalMetrics)(cell.stateWhileLoopEvidence, cell.stateDropoutEvidence)
  }

  protected def createModel[S, SS](
      name: String,
      languages: Seq[(Language, Vocabulary)],
      dataConfig: DataConfig,
      env: Environment,
      parameterManager: ParameterManager,
      cell: Cell[S, SS],
      numUnits: Int,
      residual: Boolean,
      dropout: Option[Float],
      attention: Boolean,
      labelSmoothing: Float,
      summarySteps: Int,
      checkpointSteps: Int,
      beamWidth: Int,
      lengthPenaltyWeight: Float,
      decoderMaxLengthFactor: Float,
      optConfig: Model.OptConfig,
      logConfig: Model.LogConfig,
      evalDatasets: Seq[(String, FileParallelDataset)],
      evalMetrics: Seq[MTMetric]
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): RNNModel[S, SS]

  protected def cellFromString(cellString: String): Cell[_, _] = {
    val parts = cellString.split(":")
    val activation: Output => Output = {
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
      case "lstm" if parts.length == 3 => BasicLSTM(parts(2).toFloat, activation)
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
        case _ => throw new IllegalArgumentException(s"'$value' does not represent a valid model architecture.")
      }
    })
  }
}

case class RNN(
    numEncoderLayers: Int = 2,
    numDecoderLayers: Int = 2
) extends ModelArchitecture {
  override val name: String = "rnn"

  override protected def createModel[S, SS](
      name: String,
      languages: Seq[(Language, Vocabulary)],
      dataConfig: DataConfig,
      env: Environment,
      parameterManager: ParameterManager,
      cell: Cell[S, SS],
      numUnits: Int,
      residual: Boolean,
      dropout: Option[Float],
      attention: Boolean,
      labelSmoothing: Float,
      summarySteps: Int,
      checkpointSteps: Int,
      beamWidth: Int,
      lengthPenaltyWeight: Float,
      decoderMaxLengthFactor: Float,
      optConfig: Model.OptConfig,
      logConfig: Model.LogConfig,
      evalDatasets: Seq[(String, FileParallelDataset)],
      evalMetrics: Seq[MTMetric]
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): RNNModel[S, SS] = {
    RNNModel(
      name = name,
      languages = languages,
      dataConfig = dataConfig,
      config = RNNModel.Config(
        env,
        parameterManager,
        UnidirectionalRNNEncoder(
          cell = cell,
          numUnits = numUnits,
          numLayers = numEncoderLayers,
          residual = residual,
          dropout = dropout),
        UnidirectionalRNNDecoder(
          cell = cell,
          numUnits = numUnits,
          numLayers = numDecoderLayers,
          residual = residual,
          dropout = dropout,
          attention = if (attention) Some(LuongRNNAttention(scaled = true)) else None,
          outputAttention = attention),
        labelSmoothing = labelSmoothing,
        summarySteps = summarySteps,
        checkpointSteps = checkpointSteps,
        beamWidth = beamWidth,
        lengthPenaltyWeight = lengthPenaltyWeight,
        decoderMaxLengthFactor = decoderMaxLengthFactor),
      optConfig = optConfig,
      logConfig = logConfig
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

  override protected def createModel[S, SS](
      name: String,
      languages: Seq[(Language, Vocabulary)],
      dataConfig: DataConfig,
      env: Environment,
      parameterManager: ParameterManager,
      cell: Cell[S, SS],
      numUnits: Int,
      residual: Boolean,
      dropout: Option[Float],
      attention: Boolean,
      labelSmoothing: Float,
      summarySteps: Int,
      checkpointSteps: Int,
      beamWidth: Int,
      lengthPenaltyWeight: Float,
      decoderMaxLengthFactor: Float,
      optConfig: Model.OptConfig,
      logConfig: Model.LogConfig,
      evalDatasets: Seq[(String, FileParallelDataset)],
      evalMetrics: Seq[MTMetric]
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): RNNModel[S, SS] = {
    RNNModel(
      name = name,
      languages = languages,
      dataConfig = dataConfig,
      config = RNNModel.Config(
        env,
        parameterManager,
        BidirectionalRNNEncoder(
          cell = cell,
          numUnits = numUnits,
          numLayers = numEncoderLayers,
          residual = residual,
          dropout = dropout),
        UnidirectionalRNNDecoder(
          cell = cell,
          numUnits = numUnits,
          numLayers = numDecoderLayers,
          residual = residual,
          dropout = dropout,
          attention = if (attention) Some(LuongRNNAttention(scaled = true)) else None,
          outputAttention = attention),
        labelSmoothing = labelSmoothing,
        summarySteps = summarySteps,
        checkpointSteps = checkpointSteps,
        beamWidth = beamWidth,
        lengthPenaltyWeight = lengthPenaltyWeight,
        decoderMaxLengthFactor = decoderMaxLengthFactor),
      optConfig = optConfig,
      logConfig = logConfig
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

  override protected def createModel[S, SS](
      name: String,
      languages: Seq[(Language, Vocabulary)],
      dataConfig: DataConfig,
      env: Environment,
      parameterManager: ParameterManager,
      cell: Cell[S, SS],
      numUnits: Int,
      residual: Boolean,
      dropout: Option[Float],
      attention: Boolean,
      labelSmoothing: Float,
      summarySteps: Int,
      checkpointSteps: Int,
      beamWidth: Int,
      lengthPenaltyWeight: Float,
      decoderMaxLengthFactor: Float,
      optConfig: Model.OptConfig,
      logConfig: Model.LogConfig,
      evalDatasets: Seq[(String, FileParallelDataset)],
      evalMetrics: Seq[MTMetric]
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): RNNModel[S, SS] = {
    RNNModel(
      name = name,
      languages = languages,
      dataConfig = dataConfig,
      config = RNNModel.Config(
        env,
        parameterManager,
        GNMTEncoder(
          cell = cell,
          numUnits = numUnits,
          numBiLayers = numBiLayers,
          numUniLayers = numUniLayers,
          numUniResLayers = numUniResLayers,
          dropout = dropout),
        GNMTDecoder(
          cell = cell,
          numUnits = numUnits,
          numLayers = numBiLayers + numUniLayers,
          numResLayers = numUniResLayers,
          attention = BahdanauRNNAttention(normalized = true),
          dropout = dropout,
          useNewAttention = attention),
        labelSmoothing = labelSmoothing,
        summarySteps = summarySteps,
        checkpointSteps = checkpointSteps,
        beamWidth = beamWidth,
        lengthPenaltyWeight = lengthPenaltyWeight,
        decoderMaxLengthFactor = decoderMaxLengthFactor),
      optConfig = optConfig,
      logConfig = logConfig
    )(
      evalDatasets = evalDatasets,
      evalMetrics = evalMetrics)
  }

  override def toString: String = s"gnmt:$numBiLayers:$numUniLayers:$numUniResLayers"
}
