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

import org.platanios.symphony.mt.Language
import org.platanios.tensorflow.api._

/**
  * @author Emmanouil Antonios Platanios
  */
trait ParametersManager[I, C] {
  protected var context: Option[C] = None

  def initialize(information: Option[I] = None): Unit

  def getContext: Option[C] = this.context
  def setContext(context: C): Unit = this.context = Some(context)
  def resetContext(): Unit = this.context = None

  def get(
      name: String,
      dataType: DataType,
      shape: Shape,
      variableInitializer: tf.VariableInitializer = null,
      variableReuse: tf.VariableReuse = tf.ReuseOrCreateNewVariable
  ): Output
}

case class DefaultParametersManager[I, C](
    variableInitializer: tf.VariableInitializer = null
) extends ParametersManager[I, C] {
  override def initialize(information: Option[I]): Unit = ()

  override def get(
      name: String,
      dataType: DataType,
      shape: Shape,
      variableInitializer: tf.VariableInitializer = variableInitializer,
      variableReuse: tf.VariableReuse = tf.ReuseOrCreateNewVariable
  ): Output = {
    tf.variable(name, dataType, shape, initializer = variableInitializer, reuse = variableReuse).value
  }
}

case class PerLanguagePairParametersManager(
    variableInitializer: tf.VariableInitializer = null
) extends ParametersManager[Seq[Language], (Output, Output)] {
  override def initialize(information: Option[Seq[Language]]): Unit = {
    ???
  }

  override def get(
      name: String,
      dataType: DataType,
      shape: Shape,
      variableInitializer: tf.VariableInitializer = variableInitializer,
      variableReuse: tf.VariableReuse = tf.ReuseOrCreateNewVariable
  ): Output = {
    tf.variable(name, dataType, shape, initializer = variableInitializer, reuse = variableReuse).value
  }
}
