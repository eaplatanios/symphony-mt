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
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class RNNEncoder[T: TF : IsNotQuantized, State]()
    extends Encoder[T, Tuple[Output[T], Seq[State]]] {
  override def create(
      config: RNNModel.Config[T, _],
      srcSequences: Output[Int],
      srcSequenceLengths: Output[Int]
  )(implicit
      stage: Stage,
      mode: Mode,
      env: Environment,
      parameterManager: ParameterManager,
      deviceManager: DeviceManager,
      context: Output[Int]
  ): Tuple[Output[T], Seq[State]]

  def embedSequences(
      config: RNNModel.Config[T, _],
      srcSequences: Output[Int],
      srcSequenceLengths: Output[Int]
  )(implicit
      parameterManager: ParameterManager,
      context: Output[Int]
  ): (Output[T], Output[Int]) = {
    val embeddedSrcSequences = parameterManager.wordEmbeddings(context(0))(context)(srcSequences)
    val (embeddedSequences, embeddedSequenceLengths) = parameterManager.postprocessEmbeddedSequences(
      context(0), context(1), embeddedSrcSequences, srcSequenceLengths)
    if (config.timeMajor)
      (embeddedSequences.transpose(Seq(1, 0, 2)).castTo[T], embeddedSequenceLengths)
    else
      (embeddedSequences.castTo[T], embeddedSequenceLengths)
  }
}
