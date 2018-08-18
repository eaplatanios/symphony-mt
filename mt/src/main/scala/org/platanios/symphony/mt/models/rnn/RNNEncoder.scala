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

import org.platanios.symphony.mt.Environment
import org.platanios.symphony.mt.models._
import org.platanios.symphony.mt.models.parameters.ParameterManager
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class RNNEncoder[S, SS]()(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) extends Encoder[Tuple[Output, Seq[S]]] {
  override def create(
      config: RNNModel.Config[_, _],
      srcLanguage: Output,
      tgtLanguage: Output,
      srcSequences: Output,
      srcSequenceLengths: Output
  )(implicit
      stage: Stage,
      mode: Mode,
      env: Environment,
      parameterManager: ParameterManager,
      deviceManager: DeviceManager,
      context: Output
  ): Tuple[Output, Seq[S]]

  def embedSequences(
      config: RNNModel.Config[_, _],
      srcLanguage: Output,
      tgtLanguage: Output,
      srcSequences: Output,
      srcSequenceLengths: Output
  )(implicit
      parameterManager: ParameterManager,
      context: Output
  ): (Output, Output) = {
    val embeddedSrcSequences = parameterManager.wordEmbeddings(srcLanguage)(context)(srcSequences)
    val (embeddedSequences, embeddedSequenceLengths) = parameterManager.postprocessEmbeddedSequences(
      srcLanguage, tgtLanguage, embeddedSrcSequences, srcSequenceLengths)
    if (config.timeMajor)
      (embeddedSequences.transpose(Seq(1, 0, 2)), embeddedSequenceLengths)
    else
      (embeddedSequences, embeddedSequenceLengths)
  }
}
