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
import org.platanios.symphony.mt.models.parameters.ParameterManager
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.learn.Mode

/**
  * @author Emmanouil Antonios Platanios
  */
trait Normalization {
  def apply[T: TF : IsNotQuantized](
      input: Output[T],
      depth: Option[Int] = None,
      epsilon: Float = 1e-12f,
      name: String = "Normalization"
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      stage: Stage,
      context: Output[Int]
  ): Output[T]
}

/** Applies no normalization to the input tensor. */
case object NoNormalization extends Normalization {
  override def apply[T: TF : IsNotQuantized](
      input: Output[T],
      depth: Option[Int] = None,
      epsilon: Float = 1e-12f,
      name: String = "Normalization"
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      stage: Stage,
      context: Output[Int]
  ): Output[T] = {
    input
  }
}

case class LayerNormalization(reuse: tf.VariableReuse = tf.ReuseOrCreateNewVariable) extends Normalization {
  override def apply[T: TF : IsNotQuantized](
      input: Output[T],
      depth: Option[Int] = None,
      epsilon: Float = 1e-12f,
      name: String = "Normalization"
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      stage: Stage,
      context: Output[Int]
  ): Output[T] = {
    val numFilters = depth.getOrElse(input.shape(-1))
    tf.variableScope(name, reuse) {
      val scale = parameterManager.get[T]("Scale", Shape(numFilters), tf.OnesInitializer)
      val bias = parameterManager.get[T]("Bias", Shape(numFilters), tf.ZerosInitializer)
      val mean = tf.mean(input, axes = -1, keepDims = true)
      val variance = tf.mean(tf.square(input - mean), axes = -1, keepDims = true)
      val normalizedInput = (input - mean) * tf.rsqrt(variance + tf.constant[Float](epsilon).castTo[T])
      normalizedInput * scale + bias
    }
  }
}

// TODO: !!!
case object BatchNormalization extends Normalization {
  override def apply[T: TF : IsNotQuantized](
      input: Output[T],
      depth: Option[Int] = None,
      epsilon: Float = 1e-12f,
      name: String = "Normalization"
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      stage: Stage,
      context: Output[Int]
  ): Output[T] = {
    ???
  }
}

case object NoamNormalization extends Normalization {
  override def apply[T: TF : IsNotQuantized](
      input: Output[T],
      depth: Option[Int] = None,
      epsilon: Float = 1e-12f,
      name: String = "Normalization"
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      stage: Stage,
      context: Output[Int]
  ): Output[T] = {
    tf.nameScope(name) {
      tf.l2Normalize(input, input.rank - 1, epsilon) * tf.sqrt(tf.constant[Float](input.shape(-1))).castTo[T]
    }
  }
}
