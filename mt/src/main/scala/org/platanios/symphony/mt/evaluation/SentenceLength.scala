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
import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.ops.metrics.Metric._

// TODO: Use weights.

/**
  * @param  variablesCollections Graph collections in which to add the metric variables (for streaming metrics).
  * @param  valuesCollections    Graph collections in which to add the metric values.
  * @param  updatesCollections   Graph collections in which to add the metric updates.
  * @param  resetsCollections    Graph collections in which to add the metric resets.
  * @param  name                 Name prefix for the created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class SentenceLength protected (
    val forHypothesis: Boolean,
    val variablesCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
    val valuesCollections: Set[Graph.Key[Output]] = Set(METRIC_VALUES),
    val updatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
    val resetsCollections: Set[Graph.Key[Op]] = Set(METRIC_RESETS),
    override val name: String = "SentenceLength"
) extends MTMetric {
  // TODO: Move to the TF metric class.
  protected def sanitize(name: String): String = name.replace(' ', '_')

  override def compute(
      values: ((Output, Output, Output), (Output, Output)),
      weights: Output = null,
      name: String = this.name
  ): Output = {
    val ((_, _, hypLen), (_, refLen)) = values
    val len = if (forHypothesis) hypLen else refLen
    var ops = Set(len.op)
    if (weights != null)
      ops += weights.op
    val sanitizedName = sanitize(name)
    tf.createWithNameScope(sanitizedName, ops) {
      tf.mean(len)
    }
  }

  override def streaming(
      values: ((Output, Output, Output), (Output, Output)),
      weights: Output,
      name: String = this.name
  ): Metric.StreamingInstance[Output] = {
    val ((_, _, hypLen), (_, refLen)) = values
    val len = if (forHypothesis) hypLen else refLen
    var ops = Set(len.op)
    if (weights != null)
      ops += weights.op
    val sanitizedName = sanitize(name)
    tf.variableScope(sanitizedName) {
      tf.createWithNameScope(sanitizedName, ops) {
        val length = variable("Length", INT32, Shape(), tf.ZerosInitializer, variablesCollections)
        val count = variable("Count", INT32, Shape(), tf.ZerosInitializer, variablesCollections)
        val updateLength = length.assignAdd(tf.sum(len))
        val updateCount = count.assignAdd(tf.size(len))
        val value = tf.divide(length.value.cast(FLOAT32), count.value, name = "Value")
        val update = tf.divide(updateLength.cast(FLOAT32), updateCount.cast(FLOAT32), name = "Update")
        val reset = tf.group(Set(length.initializer, count.initializer), name = "Reset")
        valuesCollections.foreach(tf.currentGraph.addToCollection(value, _))
        updatesCollections.foreach(tf.currentGraph.addToCollection(update, _))
        resetsCollections.foreach(tf.currentGraph.addToCollection(reset, _))
        Metric.StreamingInstance(value, update, reset, Set(length, count))
      }
    }
  }
}

object SentenceLength {
  def apply(
      forHypothesis: Boolean,
      variablesCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
      valuesCollections: Set[Graph.Key[Output]] = Set(METRIC_VALUES),
      updatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
      resetsCollections: Set[Graph.Key[Op]] = Set(METRIC_RESETS),
      name: String = "SentenceLength"
  ): SentenceLength = {
    new SentenceLength(
      forHypothesis, variablesCollections, valuesCollections, updatesCollections, resetsCollections, name)
  }
}
