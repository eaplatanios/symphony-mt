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

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.models.{Sentences, SentencesWithLanguage, SentencesWithLanguagePair}
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.ops.metrics.Metric._
import org.platanios.tensorflow.api.tensors.Tensor

// TODO: Use weights.

/** Contains methods for computing the exact-match accuracy for pairs of sequences.
  *
  * @param  variablesCollections Graph collections in which to add the metric variables (for streaming metrics).
  * @param  valuesCollections    Graph collections in which to add the metric values.
  * @param  updatesCollections   Graph collections in which to add the metric updates.
  * @param  resetsCollections    Graph collections in which to add the metric resets.
  * @param  name                 Name for this metric.
  *
  * @author Emmanouil Antonios Platanios
  */
class ExactMatchAccuracy protected (
    val variablesCollections: Set[Graph.Key[Variable[Any]]] = Set(METRIC_VARIABLES),
    val valuesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_VALUES),
    val updatesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_UPDATES),
    val resetsCollections: Set[Graph.Key[UntypedOp]] = Set(METRIC_RESETS),
    override val name: String = "ExactMatchAccuracy"
)(implicit languages: Seq[(Language, Vocabulary)]) extends MTMetric {
  // TODO: Do not ignore the weights.
  // TODO: Properly handle multiple reference sequences.

  private val endSeqToken = Tensor.fill[String](Shape())(Vocabulary.END_OF_SEQUENCE_TOKEN)

  override def compute(
      values: (SentencesWithLanguage[String], (SentencesWithLanguagePair[String], Sentences[String])),
      weights: Option[Output[Float]] = None,
      name: String = this.name
  ): Output[Float] = {
    var hypSentences = values._1._2._1
    var refSentences = values._2._2._1
    tf.nameScope(name) {
      val refLength = tf.shape(refSentences).apply(1)
      val hypLength = tf.shape(hypSentences).apply(1)
      val maxLength = tf.maximum(refLength, hypLength)
      hypSentences = tf.pad(
        hypSentences,
        Output[Int](
          Output[Int](0, 0),
          Output[Int](0, maxLength - hypLength)),
        tf.ConstantPadding(Some(endSeqToken)))
      refSentences = tf.pad(
        refSentences,
        Output[Int](
          Output[Int](0, 0),
          Output[Int](0, maxLength - refLength)),
        tf.ConstantPadding(Some(endSeqToken)))
      tf.sum(tf.all(tf.equal(hypSentences, refSentences), axes = -1).castTo[Float])
    }
  }

  override def streaming(
      values: (SentencesWithLanguage[String], (SentencesWithLanguagePair[String], Sentences[String])),
      weights: Option[Output[Float]] = None,
      name: String = this.name
  ): Metric.StreamingInstance[Output[Float]] = {
    var hypSentences = values._1._2._1
    var refSentences = values._2._2._1
    tf.variableScope(name) {
      tf.nameScope(name) {
        val refLength = tf.shape(refSentences).apply(1)
        val hypLength = tf.shape(hypSentences).apply(1)
        val maxLength = tf.maximum(refLength, hypLength)
        hypSentences = tf.pad(
          hypSentences,
          Output[Int](
            Output[Int](0, 0),
            Output[Int](0, maxLength - hypLength)),
          tf.ConstantPadding(Some(endSeqToken)))
        refSentences = tf.pad(
          refSentences,
          Output[Int](
            Output[Int](0, 0),
            Output[Int](0, maxLength - refLength)),
          tf.ConstantPadding(Some(endSeqToken)))
        val correctCount = variable[Long]("CorrectCount", Shape(), tf.ZerosInitializer, variablesCollections)
        val totalCount = variable[Long]("TotalCount", Shape(), tf.ZerosInitializer, variablesCollections)
        val _correctCount = tf.sum(tf.all(tf.equal(hypSentences, refSentences), axes = -1).castTo[Long])
        val _totalCount = tf.shape(hypSentences).apply(0).reshape(Shape()).castTo[Long]
        val updateCorrectCount = correctCount.assignAdd(_correctCount, name = "UpdateCorrectCount")
        val updateTotalCount = totalCount.assignAdd(_totalCount, name = "UpdateTotalCount")
        val value = correctCount.castTo[Float] / totalCount.castTo[Float]
        val update = updateCorrectCount.castTo[Float] / updateTotalCount.castTo[Float]
        val reset = tf.group(Set(correctCount.initializer, totalCount.initializer), name = "Reset")
        valuesCollections.foreach(tf.currentGraph.addToCollection(_)(value.asUntyped))
        updatesCollections.foreach(tf.currentGraph.addToCollection(_)(update.asUntyped))
        resetsCollections.foreach(tf.currentGraph.addToCollection(_)(reset.asUntyped))
        Metric.StreamingInstance(value, update, reset, Set(correctCount.asUntyped, totalCount.asUntyped))
      }
    }
  }
}

object ExactMatchAccuracy {
  def apply(
      variablesCollections: Set[Graph.Key[Variable[Any]]] = Set(METRIC_VARIABLES),
      valuesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_VALUES),
      updatesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_UPDATES),
      resetsCollections: Set[Graph.Key[UntypedOp]] = Set(METRIC_RESETS),
      name: String = "ExactMatchAccuracy"
  )(implicit languages: Seq[(Language, Vocabulary)]): ExactMatchAccuracy = {
    new ExactMatchAccuracy(
      variablesCollections, valuesCollections, updatesCollections, resetsCollections, name
    )(languages)
  }

  case class Counts(correctCount: Tensor[Long], totalCount: Tensor[Long])
}
