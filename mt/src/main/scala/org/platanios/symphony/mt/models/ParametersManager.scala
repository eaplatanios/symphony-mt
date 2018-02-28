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

import org.platanios.tensorflow.api._

/**
  * @author Emmanouil Antonios Platanios
  */
trait ParametersManager[I] {
  def get(
      name: String,
      dataType: DataType,
      shape: Shape,
      variableInitializer: tf.VariableInitializer = null,
      variableReuse: tf.VariableReuse = tf.ReuseOrCreateNewVariable,
      information: Option[I] = None
  ): Output

  def withInformation(information: I): ParametersManager[I]
}

case class DefaultParametersManager[I](
    variableInitializer: tf.VariableInitializer = null,
    information: Option[I] = None
) extends ParametersManager[I] {
  override def get(
      name: String,
      dataType: DataType,
      shape: Shape,
      variableInitializer: tf.VariableInitializer = variableInitializer,
      variableReuse: tf.VariableReuse = tf.ReuseOrCreateNewVariable,
      information: Option[I] = this.information
  ): Output = {
    tf.variable(name, dataType, shape, initializer = variableInitializer, reuse = variableReuse).value
  }

  override def withInformation(information: I): ParametersManager[I] = this.copy(information = Some(information))
}
