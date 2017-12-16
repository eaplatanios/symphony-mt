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

import org.platanios.tensorflow.api._

import java.nio.file.{Path, Paths}

/**
  * @author Emmanouil Antonios Platanios
  */
case class Configuration[S, SS](
    workingDir: Path = Paths.get("temp"),
    // Model
    cell: Cell[S, SS],
    numUnits: Int,
    numUniLayers: Int = 1,
    numUniResLayers: Int = 0,
    numBiLayers: Int = 0,
    numBiResLayers: Int = 0,
    dataType: DataType = FLOAT32,
    dropout: Option[Float] = None,
    residualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output),
    decoderAttention: Option[Attention] = None,
    decoderAttentionArchitecture: AttentionArchitecture = StandardAttention,
    decoderOutputAttention: Boolean = false,
    decoderMaxLengthFactor: Float = 2.0f,
    // Inference
    inferBatchSize: Int = 32,
    inferBeamWidth: Int = 1,
    inferLengthPenaltyWeight: Float = 0.0f,
    // Logging
    logLossSteps: Int = 100,
    logEvalBatchSize: Int = 512,
    logTrainEvalSteps: Int = 1000,
    logDevEvalSteps: Int = 1000,
    logTestEvalSteps: Int = 1000,
    logDevicePlacement: Boolean = false)
