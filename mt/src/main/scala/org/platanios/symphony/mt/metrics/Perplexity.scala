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

package org.platanios.symphony.mt.metrics

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.metrics.Metric

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
    val variablesCollections: Set[Graph.Key[Variable]] = Set(Metric.METRIC_VARIABLES),
    val valuesCollections: Set[Graph.Key[Output]] = Set(Metric.METRIC_VALUES),
    val updatesCollections: Set[Graph.Key[Output]] = Set(Metric.METRIC_UPDATES),
    val resetsCollections: Set[Graph.Key[Op]] = Set(Metric.METRIC_RESETS),
    override val name: String = "Perplexity"
) extends Metric[((Output, Output), (Output, Output, Output)), Output] {
  override def compute(
      values: ((Output, Output), (Output, Output, Output)),
      weights: Output = null,
      name: String = name
  ): Output = {
    tf.createWithNameScope(name) {
      val mask = tf.sequenceMask(values._1._2, tf.shape(values._1._1)(1), dataType = values._1._1.dataType)
      val loss = tf.sum(tf.sequenceLoss(
        values._1._1, values._2._2, mask, averageAcrossTimeSteps = false, averageAcrossBatch = false))
      val length = tf.sum(values._1._2).cast(values._1._1.dataType)
      tf.exp(Metric.safeScalarDiv(loss, length))
    }
  }

  override def streaming(
      values: ((Output, Output), (Output, Output, Output)),
      weights: Output = null,
      name: String = name
  ): Metric.StreamingInstance[Output] = {
    tf.createWithVariableScope(name) {
      tf.createWithNameScope(name) {
        // Create accumulator variables
        val loss = variable("Loss", values._1._1.dataType, Shape.scalar(), tf.ZerosInitializer)
        val length = variable("Length", values._1._1.dataType, Shape.scalar(), tf.ZerosInitializer)

        // Create update ops
        val mask  = tf.sequenceMask(values._1._2, tf.shape(values._1._1)(1), dataType = values._1._1.dataType)
        val updateLoss = loss.assignAdd(tf.sum(tf.sequenceLoss(
          values._1._1, values._2._2, mask, averageAcrossTimeSteps = false, averageAcrossBatch = false)))
        val updateLength = length.assignAdd(tf.sum(values._1._2).cast(values._1._1.dataType))

        // Create value ops
        val value = tf.exp(Metric.safeScalarDiv(loss.value, length.value), name = "Value")
        val update = tf.exp(Metric.safeScalarDiv(updateLoss, updateLength), name = "Update")

        // Create reset op
        val reset = tf.group(Set(loss.initializer, length.initializer))

        // Add the created ops to the relevant graph collections
        variablesCollections.foreach(tf.currentGraph.addToCollection(loss, _))
        variablesCollections.foreach(tf.currentGraph.addToCollection(length, _))
        valuesCollections.foreach(tf.currentGraph.addToCollection(value, _))
        updatesCollections.foreach(tf.currentGraph.addToCollection(update, _))
        resetsCollections.foreach(tf.currentGraph.addToCollection(reset, _))
        Metric.StreamingInstance(value, update, reset, Set(loss, length))
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
      variablesCollections: Set[Graph.Key[Variable]] = Set(Metric.METRIC_VARIABLES),
      valuesCollections: Set[Graph.Key[Output]] = Set(Metric.METRIC_VALUES),
      updatesCollections: Set[Graph.Key[Output]] = Set(Metric.METRIC_UPDATES),
      resetsCollections: Set[Graph.Key[Op]] = Set(Metric.METRIC_RESETS),
      name: String = "Perplexity"
  ): Perplexity = {
    new Perplexity(variablesCollections, valuesCollections, updatesCollections, resetsCollections, name)
  }
}
