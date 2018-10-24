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

import org.platanios.symphony.mt.Environment
import org.platanios.symphony.mt.models.Model.DecodingMode
import org.platanios.symphony.mt.models.parameters.ParameterManager
import org.platanios.symphony.mt.models.rnn.RNNDecoder
import org.platanios.tensorflow.api.core.types.TF
import org.platanios.tensorflow.api.implicits.helpers.OutputStructure
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.seq2seq.decoders.BeamSearchDecoder

/**
  * @author Emmanouil Antonios Platanios
  */
trait Decoder[T, EncoderState] {
  def create[O: TF](
      decodingMode: DecodingMode[O],
      config: RNNModel.Config[T, _],
      encoderState: EncoderState,
      beginOfSequenceToken: String,
      endOfSequenceToken: String,
      tgtSequences: Output[Int] = null,
      tgtSequenceLengths: Output[Int] = null
  )(implicit
      stage: Stage,
      mode: Mode,
      env: Environment,
      parameterManager: ParameterManager,
      deviceManager: DeviceManager,
      context: Output[Int],
  ): RNNDecoder.DecoderOutput[O]
}

object Decoder {
  def tileForBeamSearch[S: OutputStructure](value: S, beamWidth: Int): S = {
    OutputStructure[S].map(
      value,
      converter = BeamSearchDecoder.TileBatchConverter(beamWidth))
  }
}
