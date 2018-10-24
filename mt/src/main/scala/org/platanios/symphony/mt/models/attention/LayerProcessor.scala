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
import org.platanios.symphony.mt.models.helpers.Common
import org.platanios.symphony.mt.models.parameters.ParameterManager
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsHalfOrFloatOrDouble, TF}
import org.platanios.tensorflow.api.learn.Mode

/**
  * @author Emmanouil Antonios Platanios
  */
trait LayerProcessor {
  @throws[IllegalArgumentException]
  def apply[T: TF : IsHalfOrFloatOrDouble](
      value: Output[T],
      previousValue: Option[Output[T]],
      name: String = "LayerProcessor"
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      stage: Stage,
      context: Output[Int]
  ): Output[T]
}

case object AddResidualConnection extends LayerProcessor {
  @throws[IllegalArgumentException]
  override def apply[T: TF : IsHalfOrFloatOrDouble](
      value: Output[T],
      previousValue: Option[Output[T]],
      name: String = "AddResidualConnection"
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      stage: Stage,
      context: Output[Int]
  ): Output[T] = {
    previousValue match {
      case Some(v) => value + v
      case None => throw new IllegalArgumentException(
        "No residual connection can be added when no previous value is provided.")
    }
  }
}

case class Normalize(normalization: Normalization, epsilon: Float = 1e-12f) extends LayerProcessor {
  @throws[IllegalArgumentException]
  override def apply[T: TF : IsHalfOrFloatOrDouble](
      value: Output[T],
      previousValue: Option[Output[T]],
      name: String = "Normalize"
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      stage: Stage,
      context: Output[Int]
  ): Output[T] = {
    normalization(value, epsilon = epsilon, name = name)
  }
}

case class Dropout(
    dropoutRate: Float,
    scaleOutput: Boolean = true,
    broadcastAxes: Set[Int] = Set.empty
) extends LayerProcessor {
  @throws[IllegalArgumentException]
  override def apply[T: TF : IsHalfOrFloatOrDouble](
      value: Output[T],
      previousValue: Option[Output[T]],
      name: String = "Dropout"
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      stage: Stage,
      context: Output[Int]
  ): Output[T] = {
    if (mode.isTraining)
      Common.dropoutWithBroadcastAxes(value, 1.0f - dropoutRate, scaleOutput, broadcastAxes)
    else
      value
  }
}

object LayerProcessor {
  /** Applies a sequence of functions to the input of a layer.
    *
    * @param  input            Layer input.
    * @param  processors       Layer processors to apply.
    * @param  mode             Current learning mode (e.g., training or evaluation).
    * @param  parameterManager Parameter manager to use, if parameters are required.
    * @return Processed layer input.
    */
  def layerPreprocess[T: TF : IsHalfOrFloatOrDouble](
      input: Output[T],
      processors: Seq[LayerProcessor]
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      stage: Stage,
      context: Output[Int]
  ): Output[T] = {
    processors.foldLeft(input) {
      case (value, processor) => processor(value, None)
    }
  }

  /** Applies a sequence of functions to the output of a layer, potentially depending on its input.
    *
    * @param  input            Layer input.
    * @param  output           Layer output.
    * @param  processors       Layer processors to apply.
    * @param  mode             Current learning mode (e.g., training or evaluation).
    * @param  parameterManager Parameter manager to use, if parameters are required.
    * @return Processed layer output.
    */
  def layerPostprocess[T: TF : IsHalfOrFloatOrDouble](
      input: Output[T],
      output: Output[T],
      processors: Seq[LayerProcessor]
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      stage: Stage,
      context: Output[Int]
  ): Output[T] = {
    processors.foldLeft(output) {
      case (value, processor) => processor(value, Some(input))
    }
  }
}
