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

import org.platanios.symphony.mt.{Environment, Language}
import org.platanios.symphony.mt.data.{DataConfig, FileParallelDataset}
import org.platanios.symphony.mt.data.loaders._
import org.platanios.symphony.mt.experiments.{Experiment, Metric}
import org.platanios.symphony.mt.models.{Model, ModelConfig}
import org.platanios.symphony.mt.models.Transformation.{Decoder, Encoder}
import org.platanios.symphony.mt.models.parameters.{PairwiseManager, ParameterManager}
import org.platanios.symphony.mt.models.pivoting.{NoPivot, Pivot, SinglePivot}
import org.platanios.symphony.mt.models.rnn.attention.{BahdanauRNNAttention, LuongRNNAttention}
import org.platanios.symphony.mt.models.rnn._
import org.platanios.symphony.mt.models.transformer.{TransformerDecoder, TransformerEncoder}
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape, Zero}

import com.typesafe.config.Config

/**
  * @author Emmanouil Antonios Platanios
  */
class ModelConfigParser[T: TF : IsHalfOrFloatOrDouble](
    task: Experiment.Task,
    dataset: String,
    datasets: => Seq[FileParallelDataset],
    languages: => Seq[(Language, Vocabulary)],
    environment: => Environment,
    parameterManager: => ParameterManager,
    dataConfig: => DataConfig,
    modelConfig: => ModelConfig,
    name: String
) extends ConfigParser[Model[_]] {
  override def parse(config: Config): Model[_] = {
    val evalDatasets: Seq[(String, FileParallelDataset, Float)] = {
      val evalDatasetTags = config.getString("evaluation.datasets").split(',').map(dataset => {
        val parts = dataset.split(':')
        (parts(0), parts(1).toFloat)
      })
      task match {
        case Experiment.Train | Experiment.Evaluate =>
          val evalTags = dataset match {
            case "iwslt14" => evalDatasetTags.map(t => (s"IWSLT-14/${t._1}", IWSLT14Loader.Tag.fromName(t._1), t._2))
            case "iwslt15" => evalDatasetTags.map(t => (s"IWSLT-15/${t._1}", IWSLT15Loader.Tag.fromName(t._1), t._2))
            case "iwslt16" => evalDatasetTags.map(t => (s"IWSLT-16/${t._1}", IWSLT16Loader.Tag.fromName(t._1), t._2))
            case "iwslt17" => evalDatasetTags.map(t => (s"IWSLT-17/${t._1}", IWSLT17Loader.Tag.fromName(t._1), t._2))
            case "wmt16" => evalDatasetTags.map(t => (s"WMT-16/${t._1}", WMT16Loader.Tag.fromName(t._1), t._2))
            case "ted_talks" => evalDatasetTags.map(t => (s"TED-Talks/${t._1}", TEDTalksLoader.Tag.fromName(t._1), t._2))
          }
          evalTags.flatMap(t => datasets.map(d => (t._1, d.filterTags(t._2), t._3)))
        case Experiment.Translate => Seq.empty
      }
    }
    val evalMetrics = config.getString("evaluation.metrics").split(',').map(Metric.cliToMTMetric(_)(languages))
    val encoder = ModelConfigParser.encoderFromConfig[T](config.getConfig("encoder"))
    val decoder = ModelConfigParser.decoderFromConfig[T](config.getConfig("decoder"))
    new Model(
      name = name,
      languages = languages,
      encoder = encoder,
      decoder = decoder,
      env = environment,
      parameterManager = parameterManager,
      dataConfig = dataConfig,
      modelConfig = modelConfig.copy(
        inferenceConfig = modelConfig.inferenceConfig.copy(
          pivot = ModelConfigParser.pivot(parameterManager, modelConfig.languagePairs)))
    )(
      evalDatasets = evalDatasets,
      evalMetrics = evalMetrics)
  }

  override def tag(config: Config, parsedValue: => Model[_]): Option[String] = {
    val encoderConfig = config.getConfig("encoder")
    val decoderConfig = config.getConfig("decoder")

    // TODO: !!! Make this more detailed.
    val stringBuilder = new StringBuilder()
    stringBuilder.append(s"enc:${encoderConfig.getString("type")}")
    if (encoderConfig.hasPath("num-layers"))
      stringBuilder.append(s":${encoderConfig.getInt("num-layers")}")
    if (encoderConfig.hasPath("residual") && encoderConfig.getBoolean("residual"))
      stringBuilder.append(":r")
    stringBuilder.append(s".dec:${decoderConfig.getString("type")}")
    if (decoderConfig.hasPath("num-layers"))
      stringBuilder.append(s":${decoderConfig.getInt("num-layers")}")
    if (decoderConfig.hasPath("residual") && decoderConfig.getBoolean("residual"))
      stringBuilder.append(":r")
    if (decoderConfig.hasPath("use-attention") && decoderConfig.getBoolean("use-attention"))
      stringBuilder.append(":a")
    Some(stringBuilder.toString)
  }
}

object ModelConfigParser {
  private def encoderFromConfig[T: TF : IsHalfOrFloatOrDouble](encoderConfig: Config): Encoder[Any] = {
    val encoderType = encoderConfig.getString("type")
    encoderType match {
      case "rnn" =>
        val cell: Cell[T, _, _] = cellFromConfig[T](encoderConfig.getConfig("cell"))

        implicit val evOutputStructureState: OutputStructure[cell.StateType] = {
          cell.evOutputStructureState.asInstanceOf[OutputStructure[cell.StateType]]
        }

        implicit val evOutputToShapeState: OutputToShape.Aux[cell.StateType, cell.StateShapeType] = {
          cell.evOutputToShapeState.asInstanceOf[OutputToShape.Aux[cell.StateType, cell.StateShapeType]]
        }

        implicit val evZeroState: Zero.Aux[cell.StateType, cell.StateShapeType] = {
          cell.evZeroState.asInstanceOf[Zero.Aux[cell.StateType, cell.StateShapeType]]
        }

        new UnidirectionalRNNEncoder[T, cell.StateType, cell.StateShapeType](
          cell = cell.asInstanceOf[Cell[cell.DataType, cell.StateType, cell.StateShapeType]],
          numUnits = encoderConfig.getInt("num-units"),
          numLayers = encoderConfig.getInt("num-layers"),
          residual = encoderConfig.getBoolean("residual"),
          dropout = {
            if (encoderConfig.hasPath("dropout"))
              Some(encoderConfig.getDouble("dropout").toFloat)
            else
              None
          }
        ).asInstanceOf[Encoder[Any]]
      case "bi-rnn" =>
        val cell: Cell[T, _, _] = cellFromConfig[T](encoderConfig.getConfig("cell"))

        implicit val evOutputStructureState: OutputStructure[cell.StateType] = {
          cell.evOutputStructureState.asInstanceOf[OutputStructure[cell.StateType]]
        }

        implicit val evOutputToShapeState: OutputToShape.Aux[cell.StateType, cell.StateShapeType] = {
          cell.evOutputToShapeState.asInstanceOf[OutputToShape.Aux[cell.StateType, cell.StateShapeType]]
        }

        implicit val evZeroState: Zero.Aux[cell.StateType, cell.StateShapeType] = {
          cell.evZeroState.asInstanceOf[Zero.Aux[cell.StateType, cell.StateShapeType]]
        }

        new BidirectionalRNNEncoder[T, cell.StateType, cell.StateShapeType](
          cell = cell.asInstanceOf[Cell[cell.DataType, cell.StateType, cell.StateShapeType]],
          numUnits = encoderConfig.getInt("num-units"),
          numLayers = encoderConfig.getInt("num-layers"),
          residual = encoderConfig.getBoolean("residual"),
          dropout = {
            if (encoderConfig.hasPath("dropout"))
              Some(encoderConfig.getDouble("dropout").toFloat)
            else
              None
          }
        ).asInstanceOf[Encoder[Any]]
      case "gnmt" =>
        val cell: Cell[T, _, _] = cellFromConfig[T](encoderConfig.getConfig("cell"))

        implicit val evOutputStructureState: OutputStructure[cell.StateType] = {
          cell.evOutputStructureState.asInstanceOf[OutputStructure[cell.StateType]]
        }

        implicit val evOutputToShapeState: OutputToShape.Aux[cell.StateType, cell.StateShapeType] = {
          cell.evOutputToShapeState.asInstanceOf[OutputToShape.Aux[cell.StateType, cell.StateShapeType]]
        }

        implicit val evZeroState: Zero.Aux[cell.StateType, cell.StateShapeType] = {
          cell.evZeroState.asInstanceOf[Zero.Aux[cell.StateType, cell.StateShapeType]]
        }

        new GNMTEncoder[T, cell.StateType, cell.StateShapeType](
          cell = cell.asInstanceOf[Cell[cell.DataType, cell.StateType, cell.StateShapeType]],
          numUnits = encoderConfig.getInt("num-units"),
          numBiLayers = encoderConfig.getInt("num-bi-layers"),
          numUniLayers = encoderConfig.getInt("num-uni-layers"),
          numUniResLayers = encoderConfig.getInt("num-uni-res-layers"),
          dropout = {
            if (encoderConfig.hasPath("dropout"))
              Some(encoderConfig.getDouble("dropout").toFloat)
            else
              None
          }
        ).asInstanceOf[Encoder[Any]]
      case "transformer" =>
        new TransformerEncoder[T](
          numUnits = encoderConfig.getInt("num-units"),
          numLayers = encoderConfig.getInt("num-layers"),
          attentionKeysDepth = encoderConfig.getInt("attention-keys-depth"),
          attentionValuesDepth = encoderConfig.getInt("attention-values-depth"),
          attentionNumHeads = encoderConfig.getInt("attention-num-heads")
        ).asInstanceOf[Encoder[Any]]
      case _ => throw new IllegalArgumentException(s"'$encoderType' does not represent a valid encoder type.")
    }
  }

  private def decoderFromConfig[T: TF : IsHalfOrFloatOrDouble](decoderConfig: Config): Decoder[Any] = {
    val decoderType = decoderConfig.getString("type")
    decoderType match {
      case "rnn" =>
        val cell: Cell[T, _, _] = cellFromConfig[T](decoderConfig.getConfig("cell"))
        val numUnits = decoderConfig.getInt("num-units")
        val numLayers = decoderConfig.getInt("num-layers")
        val residual = decoderConfig.getBoolean("residual")
        val dropout = {
          if (decoderConfig.hasPath("dropout"))
            Some(decoderConfig.getDouble("dropout").toFloat)
          else
            None
        }
        val useAttention = {
          if (decoderConfig.hasPath("use-attention"))
            decoderConfig.getBoolean("use-attention")
          else
            false
        }

        implicit val evOutputStructureState: OutputStructure[cell.StateType] = {
          cell.evOutputStructureState.asInstanceOf[OutputStructure[cell.StateType]]
        }

        implicit val evOutputToShapeState: OutputToShape.Aux[cell.StateType, cell.StateShapeType] = {
          cell.evOutputToShapeState.asInstanceOf[OutputToShape.Aux[cell.StateType, cell.StateShapeType]]
        }

        if (useAttention) {
          new UnidirectionalRNNDecoderWithAttention(
            cell = cell.asInstanceOf[Cell[cell.DataType, cell.StateType, cell.StateShapeType]],
            numUnits = numUnits,
            numLayers = numLayers,
            residual = residual,
            dropout = dropout,
            attention = new LuongRNNAttention(
              scaled = true,
              probabilityFn = (o: Output[T]) => tf.softmax(o)),
            outputAttention = true
          ).asInstanceOf[Decoder[Any]]
        } else {
          new UnidirectionalRNNDecoder(
            cell = cell.asInstanceOf[Cell[cell.DataType, cell.StateType, cell.StateShapeType]],
            numUnits = numUnits,
            numLayers = numLayers,
            residual = residual,
            dropout = dropout
          ).asInstanceOf[Decoder[Any]]
        }
      case "gnmt" =>
        val cell: Cell[T, _, _] = cellFromConfig[T](decoderConfig.getConfig("cell"))
        val dropout = {
          if (decoderConfig.hasPath("dropout"))
            Some(decoderConfig.getDouble("dropout").toFloat)
          else
            None
        }
        val useNewAttention = {
          if (decoderConfig.hasPath("use-new-attention"))
            decoderConfig.getBoolean("use-new-attention")
          else
            false
        }

        implicit val evOutputStructureState: OutputStructure[cell.StateType] = {
          cell.evOutputStructureState.asInstanceOf[OutputStructure[cell.StateType]]
        }

        implicit val evOutputToShapeState: OutputToShape.Aux[cell.StateType, cell.StateShapeType] = {
          cell.evOutputToShapeState.asInstanceOf[OutputToShape.Aux[cell.StateType, cell.StateShapeType]]
        }

        new GNMTDecoder(
          cell = cell.asInstanceOf[Cell[cell.DataType, cell.StateType, cell.StateShapeType]],
          numUnits = decoderConfig.getInt("num-units"),
          numLayers = decoderConfig.getInt("num-layers"), // TODO: Should be equal to `numBiLayers + numUniLayers`
          numResLayers = decoderConfig.getInt("num-res-layers"),
          attention = new BahdanauRNNAttention(
            normalized = true,
            probabilityFn = (o: Output[T]) => tf.softmax(o)),
          dropout = dropout,
          useNewAttention = useNewAttention
        ).asInstanceOf[Decoder[Any]]
      case "transformer" =>
        new TransformerDecoder[T](
          numLayers = decoderConfig.getInt("num-layers"),
          attentionKeysDepth = decoderConfig.getInt("attention-keys-depth"),
          attentionValuesDepth = decoderConfig.getInt("attention-values-depth"),
          attentionNumHeads = decoderConfig.getInt("attention-num-heads"),
          useEncoderDecoderAttentionCache = {
            if (decoderConfig.hasPath("use-encoder-decoder-attention-cache"))
              decoderConfig.getBoolean("use-encoder-decoder-attention-cache")
            else
              true
          }).asInstanceOf[Decoder[Any]]
      case _ => throw new IllegalArgumentException(s"'$decoderType' does not represent a valid decoder type.")
    }
  }

  private def cellFromConfig[T: TF : IsReal](cellConfig: Config): Cell[T, _, _] = {
    // Parse the cell activation function.
    val cellActivation = cellConfig.getString("activation")
    val activation: Output[T] => Output[T] = cellActivation match {
      case "sigmoid" => tf.sigmoid(_)
      case "tanh" => tf.tanh(_)
      case "relu" => tf.relu(_)
      case "relu6" => tf.relu6(_)
      case "elu" => tf.elu(_)
      case "selu" => tf.selu(_)
      case _ => throw new IllegalArgumentException(s"'$cellActivation' does not represent a valid activation function.")
    }

    // Parse the cell type.
    val cellType = cellConfig.getString("type")
    cellType match {
      case "gru" => GRU(activation)
      case "lstm" =>
        val forgetBias = if (cellConfig.hasPath("forget-bias")) cellConfig.getDouble("forget-bias").toFloat else 1.0f
        BasicLSTM(activation, forgetBias)
      case _ => throw new IllegalArgumentException(s"'$cellType' does not represent a valid RNN cell type.")
    }
  }

  private def pivot(
      parameterManager: ParameterManager,
      languagePairs: Set[(Language, Language)]
  ): Pivot = {
    parameterManager match {
      case _: PairwiseManager => SinglePivot(Language.English, languagePairs)
      case _ => NoPivot
    }
  }
}
