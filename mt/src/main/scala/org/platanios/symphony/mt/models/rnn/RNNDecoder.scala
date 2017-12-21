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

import org.platanios.symphony.mt.models.Decoder
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class RNNDecoder[S, SS]()(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) extends Decoder[Tuple[Output, Seq[S]]] {
  def create(
      encoderTuple: Tuple[Output, Seq[S]], inputSequenceLengths: Output,
      targetSequences: Output = null, targetSequenceLengths: Output = null, mode: Mode
  ): RNNDecoder.Output
}

object RNNDecoder {
  case class Output(sequences: tf.Output, sequenceLengths: tf.Output)
}
