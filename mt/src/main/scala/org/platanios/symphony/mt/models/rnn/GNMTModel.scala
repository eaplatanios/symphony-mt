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

import org.platanios.symphony.mt.core.{Environment, Language}
import org.platanios.symphony.mt.data.{DataConfig, Vocabulary}
import org.platanios.symphony.mt.data.Datasets.MTTextLinesDataset
import org.platanios.symphony.mt.models.TrainConfig
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable

/**
  * @author Emmanouil Antonios Platanios
  */
class GNMTModel[S, SS](
    override val config: Configuration[S, SS],
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
    override val trainConfig: TrainConfig = TrainConfig(),
    override val dataConfig: DataConfig = DataConfig(),
    override val name: String = "GNMTModel"
)(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) extends Model[S, SS] {
  override protected def encoder: Encoder[S, SS] = {
    GNMTEncoder(env, config, dataConfig, srcVocabulary, "Encoder")(evS, evSDropout)
  }

  override protected def decoder: Decoder[S, SS] = {
    GNMTDecoder(env, config, dataConfig, srcVocabulary, tgtVocabulary, "Decoder")(evS, evSDropout)
  }
}

object GNMTModel {
  def apply[S, SS](
      config: Configuration[S, SS],
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
      trainConfig: TrainConfig = TrainConfig(),
      dataConfig: DataConfig = DataConfig(),
      name: String = "GNMTModel"
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): GNMTModel[S, SS] = {
    new GNMTModel[S, SS](
      config, srcLanguage, tgtLanguage, srcVocabulary, tgtVocabulary, srcTrainDataset, tgtTrainDataset, srcDevDataset,
      tgtDevDataset, srcTestDataset, tgtTestDataset, env, trainConfig, dataConfig, name)
  }
}
