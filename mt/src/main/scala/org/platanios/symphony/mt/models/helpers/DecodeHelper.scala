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

package org.platanios.symphony.mt.models.helpers

import org.platanios.tensorflow.api._

/**
  * @author Emmanouil Antonios Platanios
  */
trait DecodeHelper[EncoderOutput, CacheType] {
  def batchSize(encoderOutput: EncoderOutput): Output[Int]

  def decode(
      encoderOutput: EncoderOutput,
      decodingFn: (Output[Int], Output[Int], EncoderOutput, CacheType) => (Output[Float], CacheType),
      decodingLength: Output[Int],
      endOfSequenceID: Output[Int]
  ): DecodeHelper.Result
}

object DecodeHelper {
  case class Result(
      outputs: (Output[Int], Output[Int]),
      scores: Option[Output[Float]] = None)

  /** Returns the shape of `value`, with the inner dimensions set to unknown size. */
  def stateShapeInvariants(value: Output[_]): Shape = {
    Shape(value.shape(0)) ++ Shape.fromSeq(Seq.fill(value.rank - 2)(-1)) ++ Shape(value.shape(-1))
  }
}
