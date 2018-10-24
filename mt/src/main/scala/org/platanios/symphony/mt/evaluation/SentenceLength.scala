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

import org.platanios.symphony.mt.models.{Sentences, SentencesWithLanguage, SentencesWithLanguagePair}
import org.platanios.tensorflow.api._
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
    val variablesCollections: Set[Graph.Key[Variable[Any]]] = Set(METRIC_VARIABLES),
    val valuesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_VALUES),
    val updatesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_UPDATES),
    val resetsCollections: Set[Graph.Key[UntypedOp]] = Set(METRIC_RESETS),
    override val name: String = "SentenceLength"
) extends MTMetric {
  // TODO: Move to the TF metric class.
  protected def sanitize(name: String): String = name.replace(' ', '_')

  override def compute(
      values: (SentencesWithLanguage[String], (SentencesWithLanguagePair[String], Sentences[String])),
      weights: Option[Output[Float]] = None,
      name: String = this.name
  ): Output[Float] = {
    val srcSentenceLengths = values._1._2._2.toFloat
    val tgtSentenceLengths = values._2._2._2.toFloat
    tf.nameScope(sanitize(name)) {
      val length = if (forHypothesis) srcSentenceLengths else tgtSentenceLengths
      tf.mean(length)
    }
  }

  override def streaming(
      values: (SentencesWithLanguage[String], (SentencesWithLanguagePair[String], Sentences[String])),
      weights: Option[Output[Float]] = None,
      name: String = this.name
  ): Metric.StreamingInstance[Output[Float]] = {
    val srcSentenceLengths = values._1._2._2.toFloat
    val tgtSentenceLengths = values._2._2._2.toFloat
    val sanitizedName = sanitize(name)
    tf.variableScope(sanitizedName) {
      tf.nameScope(sanitizedName) {
        val totalLength = variable[Long]("Length", Shape(), tf.ZerosInitializer, variablesCollections)
        val count = variable[Long]("Count", Shape(), tf.ZerosInitializer, variablesCollections)
        val length = if (forHypothesis) srcSentenceLengths else tgtSentenceLengths
        val updateLength = totalLength.assignAdd(tf.sum(length).toLong)
        val updateCount = count.assignAdd(tf.size(length))
        val value = tf.divide(totalLength.value.toFloat, count.value.toFloat, name = "Value")
        val update = tf.divide(updateLength.toFloat, updateCount.toFloat, name = "Update")
        val reset = tf.group(Set(totalLength.initializer, count.initializer), name = "Reset")
        valuesCollections.foreach(tf.currentGraph.addToCollection(_)(value.asUntyped))
        updatesCollections.foreach(tf.currentGraph.addToCollection(_)(update.asUntyped))
        resetsCollections.foreach(tf.currentGraph.addToCollection(_)(reset.asUntyped))
        Metric.StreamingInstance(value, update, reset, Set(totalLength.asUntyped, count.asUntyped))
      }
    }
  }
}

object SentenceLength {
  def apply(
      forHypothesis: Boolean,
      variablesCollections: Set[Graph.Key[Variable[Any]]] = Set(METRIC_VARIABLES),
      valuesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_VALUES),
      updatesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_UPDATES),
      resetsCollections: Set[Graph.Key[UntypedOp]] = Set(METRIC_RESETS),
      name: String = "SentenceLength"
  ): SentenceLength = {
    new SentenceLength(
      forHypothesis, variablesCollections, valuesCollections, updatesCollections, resetsCollections, name)
  }
}
