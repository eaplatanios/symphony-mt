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

package org.platanios.symphony.mt.models.attention

import org.platanios.symphony.mt.models.Stage
import org.platanios.symphony.mt.models.helpers.{Common, PadRemover}
import org.platanios.symphony.mt.models.parameters.ParameterManager
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode

import scala.language.postfixOps

/**
  * @author Emmanouil Antonios Platanios
  */
trait FeedForwardLayer {
  def apply(
      input: Output[Float],
      paddingRemover: Option[PadRemover]
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      stage: Stage,
      context: Output[Int]
  ): Output[Float]
}

class DenseReLUDenseFeedForwardLayer protected (
    val filterSize: Int,
    val outputSize: Int,
    val reluDropoutRate: Float = 0.0f,
    val reluDropoutBroadcastAxes: Set[Int] = Set.empty,
    val name: String = "DenseReLUDense"
) extends FeedForwardLayer {
  override def apply(
      input: Output[Float],
      paddingRemover: Option[PadRemover]
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      stage: Stage,
      context: Output[Int]
  ): Output[Float] = {
    val inputShape = tf.shape(input).toInt
    val processedInput = paddingRemover.map(pr => {
      // Collapse `input` across examples.
      val collapsedShape = tf.concatenate[Int](Seq(Tensor(-1), inputShape(2 ::).toInt), axis = 0)
      val collapsedInput = tf.reshape(input, collapsedShape)
      // Remove the padding positions.
      tf.expandDims(pr.remove(collapsedInput), axis = 0)
    }).getOrElse(input)
    val output = Common.denseReLUDense(
      processedInput, filterSize, outputSize,
      reluDropoutRate, reluDropoutBroadcastAxes, name)
    paddingRemover.map(pr => {
      // Restore `output` to the original shape of `input`, including padding.
      tf.reshape(pr.restore(tf.squeeze(output, axes = Seq(0))), inputShape)
    }).getOrElse(output)
  }
}

object DenseReLUDenseFeedForwardLayer {
  def apply(
      filterSize: Int,
      outputSize: Int,
      reluDropoutRate: Float = 0.0f,
      reluDropoutBroadcastAxes: Set[Int] = Set.empty,
      name: String = "DenseReLUDense"
  ): DenseReLUDenseFeedForwardLayer = {
    new DenseReLUDenseFeedForwardLayer(filterSize, outputSize, reluDropoutRate, reluDropoutBroadcastAxes, name)
  }
}
