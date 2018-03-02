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
import org.platanios.tensorflow.api.learn.TRAINING

import scala.collection.mutable

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

case class LanguageEmbeddingsPairParametersManager(
    languageEmbeddingsSize: Int,
    variableInitializer: tf.VariableInitializer = null
) extends ParametersManager[Seq[Language], (Output, Output)] {
  protected var languageEmbeddings: Output                      = _
  protected val parameters        : mutable.Map[String, Output] = mutable.HashMap.empty[String, Output]

  override def initialize(information: Option[Seq[Language]]): Unit = {
    val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
    languageEmbeddings = tf.variable(
      s"LanguageEmbeddings", FLOAT32, Shape(information.get.length, languageEmbeddingsSize),
      initializer = embeddingsInitializer).value
    parameters.clear()
  }

  override def get(
      name: String,
      dataType: DataType,
      shape: Shape,
      variableInitializer: tf.VariableInitializer = variableInitializer,
      variableReuse: tf.VariableReuse = tf.ReuseOrCreateNewVariable
  ): Output = {
    val variableScopeName = tf.currentVariableScope.name
    val fullName = if (variableScopeName != null && variableScopeName != "") s"$variableScopeName/$name" else name

    def create(): Output = tf.createWithVariableScope(name) {
      val embeddings = languageEmbeddings.gather(tf.stack(Seq(context.get._1, context.get._2))).reshape(Shape(1, -1))
      val weights = tf.variable("Dense/Weights", FLOAT32, Shape(2 * languageEmbeddingsSize, shape.numElements.toInt))
      val bias = tf.variable("Dense/Bias", FLOAT32, Shape(shape.numElements.toInt))
      val parameters = tf.linear(embeddings, weights, bias, "Dense")
      parameters.cast(dataType).reshape(shape)
    }

    variableReuse match {
      case tf.ReuseExistingVariableOnly => parameters(fullName)
      case tf.CreateNewVariableOnly =>
        // TODO: Kind of hacky.
        val created = create()
        parameters += created.op.inputs(0).name -> created
        created
      case tf.ReuseOrCreateNewVariable => parameters.getOrElseUpdate(fullName, create())
    }
  }
}
