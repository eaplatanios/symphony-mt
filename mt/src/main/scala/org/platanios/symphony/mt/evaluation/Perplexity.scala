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

package org.platanios.symphony.mt.evaluation

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsDecimal, TF}
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.ops.metrics.Metric._

/** Perplexity metric, implemented as in the [TensorFlow NMT package](https://github.com/tensorflow/nmt).
  *
  * @param  variablesCollections Graph collections in which to add the metric variables (for streaming metrics).
  * @param  valuesCollections    Graph collections in which to add the metric values.
  * @param  updatesCollections   Graph collections in which to add the metric updates.
  * @param  resetsCollections    Graph collections in which to add the metric resets.
  * @param  name                 Name prefix for the created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class Perplexity(
    val variablesCollections: Set[Graph.Key[Variable[Any]]] = Set(METRIC_VARIABLES),
    val valuesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_VALUES),
    val updatesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_UPDATES),
    val resetsCollections: Set[Graph.Key[UntypedOp]] = Set(METRIC_RESETS),
    override val name: String = "Perplexity"
) extends Metric[((Output[Float], Output[Int]), (Output[Float], Output[Int])), Output[Float]] {
  override def compute(
      values: ((Output[Float], Output[Int]), (Output[Float], Output[Int])),
      weights: Option[Output[Float]] = None,
      name: String = name
  ): Output[Float] = {
    tf.nameScope(name) {
      val mask = tf.sequenceMask(
        lengths = values._1._2,
        maxLength = tf.shape(values._1._1).slice(1).toInt)
      val loss = tf.sum(tf.sequenceLoss[Float, Float](
        logits = values._1._1,
        labels = values._2._1,
        lossFn = tf.softmaxCrossEntropy[Float](_, _)(TF[Float], IsDecimal[Float]),
        weights = mask.toFloat,
        averageAcrossTimeSteps = false,
        averageAcrossBatch = false))
      val length = tf.sum(values._1._2)
      tf.exp(Metric.safeScalarDiv(loss, length))
    }
  }

  override def streaming(
      values: ((Output[Float], Output[Int]), (Output[Float], Output[Int])),
      weights: Option[Output[Float]] = None,
      name: String = name
  ): Metric.StreamingInstance[Output[Float]] = {
    tf.variableScope(name) {
      tf.nameScope(name) {
        // Create accumulator variables
        val totalLoss = tf.variable[Float]("Loss", Shape.scalar(), tf.ZerosInitializer)
        val length = tf.variable[Long]("Length", Shape.scalar(), tf.ZerosInitializer)

        // Create update ops
        val mask = tf.sequenceMask(
          lengths = values._1._2,
          maxLength = tf.shape(values._1._1).slice(1).toInt)
        val loss = tf.sum(tf.sequenceLoss[Float, Float](
          logits = values._1._1,
          labels = values._2._1,
          lossFn = tf.softmaxCrossEntropy[Float](_, _)(TF[Float], IsDecimal[Float]),
          weights = mask.toFloat,
          averageAcrossTimeSteps = false,
          averageAcrossBatch = false))
        val updateLoss = totalLoss.assignAdd(loss)
        val updateLength = length.assignAdd(tf.sum(values._1._2).toLong)

        // Create value ops
        val value = tf.exp(Metric.safeScalarDiv(totalLoss.value, length.value.toFloat), name = "Value")
        val update = tf.exp(Metric.safeScalarDiv(updateLoss, updateLength.toFloat), name = "Update")

        // Create reset op
        val reset = tf.group(Set(totalLoss.initializer, length.initializer))

        // Add the created ops to the relevant graph collections
        variablesCollections.foreach(tf.currentGraph.addToCollection(_)(totalLoss.asUntyped))
        variablesCollections.foreach(tf.currentGraph.addToCollection(_)(length.asUntyped))
        valuesCollections.foreach(tf.currentGraph.addToCollection(_)(value.asUntyped))
        updatesCollections.foreach(tf.currentGraph.addToCollection(_)(update.asUntyped))
        resetsCollections.foreach(tf.currentGraph.addToCollection(_)(reset.asUntyped))
        Metric.StreamingInstance(value, update, reset, Set(totalLoss.asUntyped, length.asUntyped))
      }
    }
  }
}

object Perplexity {
  /** Creates a new perplexity metric.
    *
    * @param  variablesCollections Graph collections in which to add the metric variables (for streaming metrics).
    * @param  valuesCollections    Graph collections in which to add the metric values.
    * @param  updatesCollections   Graph collections in which to add the metric updates.
    * @param  resetsCollections    Graph collections in which to add the metric resets.
    * @param  name                 Name prefix for the created ops.
    * @return New mean metric.
    */
  def apply(
      variablesCollections: Set[Graph.Key[Variable[Any]]] = Set(METRIC_VARIABLES),
      valuesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_VALUES),
      updatesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_UPDATES),
      resetsCollections: Set[Graph.Key[UntypedOp]] = Set(METRIC_RESETS),
      name: String = "Perplexity"
  ): Perplexity = {
    new Perplexity(variablesCollections, valuesCollections, updatesCollections, resetsCollections, name)
  }
}
