/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.symphony.mt.models.rnn

import org.platanios.symphony.mt.{Environment, Language, LogConfig}
import org.platanios.symphony.mt.data.Datasets.MTTextLinesDataset
import org.platanios.symphony.mt.data.{DataConfig, Vocabulary}
import org.platanios.symphony.mt.models.{InferConfig, TrainConfig}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.{Layer, LayerInstance}
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.cell.{DropoutWrapper, Tuple}

/**
  * @author Emmanouil Antonios Platanios
  */
class BasicModel[S, SS](
    val config: BasicModel.Config[S, SS],
    override val srcLanguage: Language,
    override val tgtLanguage: Language,
    override val srcVocabulary: Vocabulary,
    override val tgtVocabulary: Vocabulary,
    override val srcTrainDataset: MTTextLinesDataset,
    override val tgtTrainDataset: MTTextLinesDataset,
    override val srcDevDataset: MTTextLinesDataset = null,
    override val tgtDevDataset: MTTextLinesDataset = null,
    override val srcTestDataset: MTTextLinesDataset = null,
    override val tgtTestDataset: MTTextLinesDataset = null,
    override val env: Environment = Environment(),
    override val rnnConfig: RNNConfig = RNNConfig(),
    override val dataConfig: DataConfig = DataConfig(),
    override val trainConfig: TrainConfig = TrainConfig(),
    override val inferConfig: InferConfig = InferConfig(),
    override val logConfig: LogConfig = LogConfig(),
    override val name: String = "BasicModel"
)(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: DropoutWrapper.Supported[S]
) extends Model[S, SS] {
  override protected def encoder: Layer[(Output, Output), Tuple[Output, Seq[S]]] = {
    new Layer[(Output, Output), Tuple[Output, Seq[S]]]("Encoder") {
      override val layerType: String = "BasicEncoder"

      override protected def forward(
          input: (Output, Output),
          mode: Mode
      ): LayerInstance[(Output, Output), Tuple[Output, Seq[S]]] = {
        val tuple = config.encoder.create(input._1, input._2, mode)
        LayerInstance(input, tuple)
      }
    }
  }

  override protected def trainDecoder: Layer[
      (((Output, Output), (Output, Output, Output)), Tuple[Output, Seq[S]]), (Output, Output)] = {
    new Layer[(((Output, Output), (Output, Output, Output)), Tuple[Output, Seq[S]]), (Output, Output)]("Decoder") {
      override val layerType: String = "BasicTrainDecoder"

      override protected def forward(
          input: (((Output, Output), (Output, Output, Output)), Tuple[Output, Seq[S]]),
          mode: Mode
      ): LayerInstance[(((Output, Output), (Output, Output, Output)), Tuple[Output, Seq[S]]), (Output, Output)] = {
        val decoderInstance = config.decoder.create(input._2, input._1._1._2, input._1._2._2, input._1._2._3, mode)
        LayerInstance(input, (decoderInstance.sequences, decoderInstance.sequenceLengths))
      }
    }
  }

  override protected def inferDecoder: Layer[((Output, Output), Tuple[Output, Seq[S]]), (Output, Output)] = {
    new Layer[((Output, Output), Tuple[Output, Seq[S]]), (Output, Output)]("Decoder") {
      override val layerType: String = "BasicInferDecoder"

      override protected def forward(
          input: ((Output, Output), Tuple[Output, Seq[S]]),
          mode: Mode
      ): LayerInstance[((Output, Output), Tuple[Output, Seq[S]]), (Output, Output)] = {
        val decoderInstance = config.decoder.create(input._2, input._1._2, null, null, mode)
        LayerInstance(input, (decoderInstance.sequences, decoderInstance.sequenceLengths))
      }
    }
  }
}

object BasicModel {
  def apply[S, SS](
      config: Config[S, SS],
      srcLanguage: Language,
      tgtLanguage: Language,
      srcVocabulary: Vocabulary,
      tgtVocabulary: Vocabulary,
      srcTrainDataset: MTTextLinesDataset,
      tgtTrainDataset: MTTextLinesDataset,
      srcDevDataset: MTTextLinesDataset = null,
      tgtDevDataset: MTTextLinesDataset = null,
      srcTestDataset: MTTextLinesDataset = null,
      tgtTestDataset: MTTextLinesDataset = null,
      env: Environment = Environment(),
      rnnConfig: RNNConfig = RNNConfig(),
      dataConfig: DataConfig = DataConfig(),
      trainConfig: TrainConfig = TrainConfig(),
      inferConfig: InferConfig = InferConfig(),
      logConfig: LogConfig = LogConfig(),
      name: String = "BasicModel"
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): BasicModel[S, SS] = {
    new BasicModel[S, SS](
      config, srcLanguage, tgtLanguage, srcVocabulary, tgtVocabulary, srcTrainDataset, tgtTrainDataset, srcDevDataset,
      tgtDevDataset, srcTestDataset, tgtTestDataset, env, rnnConfig, dataConfig, trainConfig, inferConfig, logConfig,
      name)
  }

  case class Config[S, SS](encoder: Encoder[S, SS], decoder: Decoder[S, SS])
}
