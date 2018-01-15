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

package org.platanios.symphony.mt.models.rnn

import org.platanios.symphony.mt.{Environment, Language}
import org.platanios.symphony.mt.data.{DataConfig, Vocabulary}
import org.platanios.symphony.mt.models.{InferConfig, StateBasedModel}
import org.platanios.symphony.mt.models.attention.{Attention, LuongAttention}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable

/**
  * @author Emmanouil Antonios Platanios
  */
case class GNMTConfig[S, SS](
    srcLanguage: Language,
    tgtLanguage: Language,
    srcVocabulary: Vocabulary,
    tgtVocabulary: Vocabulary,
    env: Environment,
    dataConfig: DataConfig,
    inferConfig: InferConfig,
    cell: Cell[S, SS],
    numUnits: Int,
    numBiLayers: Int,
    numUniLayers: Int,
    numUniResLayers: Int,
    dataType: DataType = FLOAT32,
    dropout: Option[Float] = None,
    encoderResidualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output),
    attention: Attention = LuongAttention(scaled = true),
    useNewAttention: Boolean = false,
    override val timeMajor: Boolean = false
)(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) extends StateBasedModel.Config[S, SS](
  GNMTEncoder[S, SS](
    srcLanguage, srcVocabulary, env, cell, numUnits, numBiLayers, numUniLayers, numUniResLayers,
    dataType, dropout, encoderResidualFn)(evS, evSDropout),
  GNMTDecoder[S, SS](
    tgtLanguage, tgtVocabulary, env, dataConfig, inferConfig, cell, numUnits, numUniLayers + numBiLayers,
    numUniResLayers, dataType, dropout, attention, useNewAttention)(evS, evSDropout),
  timeMajor)
