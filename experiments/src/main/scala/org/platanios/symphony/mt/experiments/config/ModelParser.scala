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
import org.platanios.symphony.mt.config.EvaluationConfig
import org.platanios.symphony.mt.data.{DataConfig, FileParallelDataset}
import org.platanios.symphony.mt.data.loaders._
import org.platanios.symphony.mt.experiments.{Experiment, Metric}
import org.platanios.symphony.mt.models.Model
import org.platanios.symphony.mt.models.Transformation.{Decoder, Encoder}
import org.platanios.symphony.mt.models.decoders.{OutputLayer, ProjectionToWordEmbeddings, ProjectionToWords}
import org.platanios.symphony.mt.models.parameters.{PairwiseManager, ParameterManager}
import org.platanios.symphony.mt.models.pivoting.{NoPivot, Pivot, SinglePivot}
import org.platanios.symphony.mt.models.rnn.attention.{BahdanauRNNAttention, LuongRNNAttention}
import org.platanios.symphony.mt.models.rnn._
import org.platanios.symphony.mt.models.transformer.{TransformerDecoder, TransformerEncoder}
import org.platanios.symphony.mt.models.transformer.helpers.{DenseReLUDenseFeedForwardLayer, DotProductAttention}
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape, Zero}

import com.typesafe.config.Config

/**
  * @author Emmanouil Antonios Platanios
  */
class ModelParser[T: TF : IsHalfOrFloatOrDouble](
    task: Experiment.Task,
    dataset: String,
    datasets: => Seq[FileParallelDataset],
    languages: => Seq[(Language, Vocabulary)],
    environment: => Environment,
    parameterManager: => ParameterManager,
    dataConfig: => DataConfig,
    name: String
) extends ConfigParser[Model[_]] {
  protected val trainingConfigParser: TrainingConfigParser = {
    new TrainingConfigParser(datasets, dataConfig)
  }

  @throws[IllegalArgumentException]
  override def parse(config: Config): Model[_] = {
    val trainingConfig = trainingConfigParser.parse(config.get[Config]("training"))
    val evalLanguagePairs = {
      val providedPairs = Experiment.parseLanguagePairs(config.get[String]("evaluation.languages"))
      if (providedPairs.isEmpty) trainingConfig.languagePairs else providedPairs
    }
    val evalDatasets: Seq[(String, FileParallelDataset)] = {
      val evalDatasetTags = config.get[String]("evaluation.datasets").split(',')
      task match {
        case Experiment.Train | Experiment.Evaluate =>
          val evalTags = dataset match {
            case "iwslt14" => evalDatasetTags.map(t => (s"IWSLT-14/$t", IWSLT14Loader.Tag.fromName(t)))
            case "iwslt15" => evalDatasetTags.map(t => (s"IWSLT-15/$t", IWSLT15Loader.Tag.fromName(t)))
            case "iwslt16" => evalDatasetTags.map(t => (s"IWSLT-16/$t", IWSLT16Loader.Tag.fromName(t)))
            case "iwslt17" => evalDatasetTags.map(t => (s"IWSLT-17/$t", IWSLT17Loader.Tag.fromName(t)))
            case "wmt16" => evalDatasetTags.map(t => (s"WMT-16/$t", WMT16Loader.Tag.fromName(t)))
            case "ted_talks" => evalDatasetTags.map(t => (s"TED-Talks/$t", TEDTalksLoader.Tag.fromName(t)))
          }
          evalTags.flatMap(t => datasets.map(d => (t._1, d.filterTags(t._2))))
        case Experiment.Translate => Seq.empty
      }
    }
    val evalMetrics = config.get[String]("evaluation.metrics").split(',').map(Metric.cliToMTMetric(_)(languages))
    val encoder = ModelParser.encoderFromConfig[T](config.get[Config]("model.encoder"))
    val decoder = ModelParser.decoderFromConfig[T](config.get[Config]("model.decoder"))
    new Model(
      name = name,
      encoder = encoder,
      decoder = decoder,
      languages = languages,
      env = environment,
      parameterManager = parameterManager,
      dataConfig = dataConfig,
      trainingConfig = trainingConfig,
      inferenceConfig = InferenceConfigParser.parse(config.get[Config]("inference")).copy(
        pivot = ModelParser.pivot(parameterManager, trainingConfig.languagePairs)),
      evaluationConfig = EvaluationConfig(
        frequency = config.get[Int]("evaluation.frequency"),
        metrics = evalMetrics,
        datasets = evalDatasets,
        languagePairs = evalLanguagePairs))
  }

  override def tag(config: Config, parsedValue: => Model[_]): Option[String] = {
    val encoderConfig = config.get[Config]("model.encoder")
    val decoderConfig = config.get[Config]("model.decoder")

    // TODO: !!! Make this more detailed.
    val stringBuilder = new StringBuilder()
    stringBuilder.append(s"enc:${encoderConfig.get[String]("type")}")
    if (encoderConfig.hasPath("num-layers"))
      stringBuilder.append(s":${encoderConfig.get[Int]("num-layers")}")
    if (encoderConfig.hasPath("residual") && encoderConfig.get[Boolean]("residual"))
      stringBuilder.append(":r")
    if (encoderConfig.get[Boolean]("remove-first-layer-residual-connection", false))
      stringBuilder.append(":no-first-residual")
    stringBuilder.append(s".dec:${decoderConfig.get[String]("type")}")
    if (decoderConfig.hasPath("num-layers"))
      stringBuilder.append(s":${decoderConfig.get[Int]("num-layers")}")
    if (decoderConfig.hasPath("residual") && decoderConfig.get[Boolean]("residual"))
      stringBuilder.append(":r")
    if (decoderConfig.hasPath("use-attention") && decoderConfig.get[Boolean]("use-attention"))
      stringBuilder.append(":a")
    if (decoderConfig.get[Boolean]("remove-first-layer-residual-connection", false))
      stringBuilder.append(":no-first-residual")
    stringBuilder.append(s".${trainingConfigParser.tag(config.get[Config]("training"), parsedValue.trainingConfig).get}")

    Some(stringBuilder.toString)
  }
}

object ModelParser {
  @throws[IllegalArgumentException]
  private def encoderFromConfig[T: TF : IsHalfOrFloatOrDouble](encoderConfig: Config): Encoder[Any] = {
    val encoderType = encoderConfig.get[String]("type")
    encoderType match {
      case "rnn" =>
        val cell: Cell[T, _, _] = cellFromConfig[T](encoderConfig.get[Config]("cell"))

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
          numUnits = encoderConfig.get[Int]("num-units"),
          numLayers = encoderConfig.get[Int]("num-layers"),
          residual = encoderConfig.get[Boolean]("residual", default = true),
          dropout = encoderConfig.get[Float]("dropout", default = 0.0f)
        ).asInstanceOf[Encoder[Any]]
      case "bi-rnn" =>
        val cell: Cell[T, _, _] = cellFromConfig[T](encoderConfig.get[Config]("cell"))

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
          numUnits = encoderConfig.get[Int]("num-units"),
          numLayers = encoderConfig.get[Int]("num-layers"),
          residual = encoderConfig.get[Boolean]("residual", default = true),
          dropout = encoderConfig.get[Float]("dropout", default = 0.0f)
        ).asInstanceOf[Encoder[Any]]
      case "gnmt" =>
        val cell: Cell[T, _, _] = cellFromConfig[T](encoderConfig.get[Config]("cell"))

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
          numUnits = encoderConfig.get[Int]("num-units"),
          numBiLayers = encoderConfig.get[Int]("num-bi-layers"),
          numUniLayers = encoderConfig.get[Int]("num-uni-layers"),
          numUniResLayers = encoderConfig.get[Int]("num-uni-res-layers"),
          dropout = encoderConfig.get[Float]("dropout", default = 0.0f)
        ).asInstanceOf[Encoder[Any]]
      case "transformer" =>
        val numUnits = encoderConfig.get[Int]("num-units")
        new TransformerEncoder[T](
          numUnits = numUnits,
          numLayers = encoderConfig.get[Int]("num-layers"),
          useSelfAttentionProximityBias = encoderConfig.get[Boolean]("use-self-attention-proximity-bias", default = false),
          postPositionEmbeddingsDropout = encoderConfig.get[Float]("post-position-embeddings-dropout"),
          removeFirstLayerResidualConnection = encoderConfig.get[Boolean]("remove-first-layer-residual-connection", false),
          attentionKeysDepth = encoderConfig.get[Int]("attention-keys-depth"),
          attentionValuesDepth = encoderConfig.get[Int]("attention-values-depth"),
          attentionNumHeads = encoderConfig.get[Int]("attention-num-heads"),
          selfAttention = DotProductAttention(
            dropoutRate = encoderConfig.get[Float]("dot-product-attention-dropout", default = 0.1f),
            dropoutBroadcastAxes = Set.empty,
            name = "DotProductAttention"),
          feedForwardLayer = DenseReLUDenseFeedForwardLayer(
            encoderConfig.get[Int]("feed-forward-filter-size"),
            numUnits,
            encoderConfig.get[Float]("feed-forward-relu-dropout"),
            Set.empty, "FeedForward")
        ).asInstanceOf[Encoder[Any]]
      case _ => throw new IllegalArgumentException(s"'$encoderType' does not represent a valid encoder type.")
    }
  }

  @throws[IllegalArgumentException]
  private def decoderFromConfig[T: TF : IsHalfOrFloatOrDouble](decoderConfig: Config): Decoder[Any] = {
    val decoderType = decoderConfig.get[String]("type")
    decoderType match {
      case "rnn" =>
        val cell: Cell[T, _, _] = cellFromConfig[T](decoderConfig.get[Config]("cell"))
        val numUnits = decoderConfig.get[Int]("num-units")
        val numLayers = decoderConfig.get[Int]("num-layers")
        val residual = decoderConfig.get[Boolean]("residual", default = true)
        val dropout = decoderConfig.get[Float]("dropout", default = 0.0f)
        val useAttention = decoderConfig.get[Boolean]("use-attention", default = false)

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
            outputAttention = true,
            outputLayer = outputLayerFromConfig(decoderConfig)
          ).asInstanceOf[Decoder[Any]]
        } else {
          new UnidirectionalRNNDecoder(
            cell = cell.asInstanceOf[Cell[cell.DataType, cell.StateType, cell.StateShapeType]],
            numUnits = numUnits,
            numLayers = numLayers,
            residual = residual,
            dropout = dropout,
            outputLayer = outputLayerFromConfig(decoderConfig)
          ).asInstanceOf[Decoder[Any]]
        }
      case "gnmt" =>
        val cell: Cell[T, _, _] = cellFromConfig[T](decoderConfig.get[Config]("cell"))
        val dropout = decoderConfig.get[Float]("dropout", default = 0.0f)
        val useNewAttention = decoderConfig.get[Boolean]("use-new-attention", default = false)

        implicit val evOutputStructureState: OutputStructure[cell.StateType] = {
          cell.evOutputStructureState.asInstanceOf[OutputStructure[cell.StateType]]
        }

        implicit val evOutputToShapeState: OutputToShape.Aux[cell.StateType, cell.StateShapeType] = {
          cell.evOutputToShapeState.asInstanceOf[OutputToShape.Aux[cell.StateType, cell.StateShapeType]]
        }

        new GNMTDecoder(
          cell = cell.asInstanceOf[Cell[cell.DataType, cell.StateType, cell.StateShapeType]],
          numUnits = decoderConfig.get[Int]("num-units"),
          numLayers = decoderConfig.get[Int]("num-layers"), // TODO: Should be equal to `numBiLayers + numUniLayers`
          numResLayers = decoderConfig.get[Int]("num-res-layers"),
          attention = new BahdanauRNNAttention(
            normalized = true,
            probabilityFn = (o: Output[T]) => tf.softmax(o)),
          dropout = dropout,
          useNewAttention = useNewAttention,
          outputLayer = outputLayerFromConfig(decoderConfig)
        ).asInstanceOf[Decoder[Any]]
      case "transformer" =>
        new TransformerDecoder[T](
          numLayers = decoderConfig.get[Int]("num-layers"),
          useSelfAttentionProximityBias = decoderConfig.get[Boolean]("use-self-attention-proximity-bias", default = false),
          postPositionEmbeddingsDropout = decoderConfig.get[Float]("post-position-embeddings-dropout"),
          attentionKeysDepth = decoderConfig.get[Int]("attention-keys-depth"),
          attentionValuesDepth = decoderConfig.get[Int]("attention-values-depth"),
          attentionNumHeads = decoderConfig.get[Int]("attention-num-heads"),
          selfAttention = DotProductAttention(
            dropoutRate = decoderConfig.get[Float]("dot-product-attention-dropout", default = 0.1f),
            dropoutBroadcastAxes = Set.empty,
            name = "DotProductAttention"),
          feedForwardLayer = DenseReLUDenseFeedForwardLayer(
            decoderConfig.get[Int]("feed-forward-filter-size"),
            decoderConfig.get[Int]("num-units"),
            decoderConfig.get[Float]("feed-forward-relu-dropout"),
            Set.empty, "FeedForward"),
          removeFirstLayerResidualConnection = decoderConfig.get[Boolean]("remove-first-layer-residual-connection", false),
          useEncoderDecoderAttentionCache = decoderConfig.get[Boolean]("use-encoder-decoder-attention-cache", default = true),
          outputLayer = outputLayerFromConfig(decoderConfig)
        ).asInstanceOf[Decoder[Any]]
      case _ => throw new IllegalArgumentException(s"'$decoderType' does not represent a valid decoder type.")
    }
  }

  @throws[IllegalArgumentException]
  private def cellFromConfig[T: TF : IsReal](cellConfig: Config): Cell[T, _, _] = {
    // Parse the cell activation function.
    val cellActivation = cellConfig.get[String]("activation")
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
    val cellType = cellConfig.get[String]("type")
    cellType match {
      case "gru" => GRU(activation)
      case "lstm" =>
        val forgetBias = if (cellConfig.hasPath("forget-bias")) cellConfig.get[Double]("forget-bias").toFloat else 1.0f
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

  @throws[IllegalArgumentException]
  private def outputLayerFromConfig(decoderConfig: Config): OutputLayer = {
    val outputLayer = decoderConfig.get[String]("output-layer")
    outputLayer match {
      case "projection-to-words" => ProjectionToWords
      case "projection-to-word-embeddings" => ProjectionToWordEmbeddings
      case _ => throw new IllegalArgumentException(s"'$outputLayer' does not represent a valid output layer.")
    }
  }
}
