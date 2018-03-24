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
import org.platanios.symphony.mt.utilities.Encoding
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.ops.metrics.Metric._

import edu.cmu.meteor.scorer.MeteorConfiguration
import edu.cmu.meteor.scorer.MeteorScorer
import edu.cmu.meteor.scorer.MeteorStats

import scala.collection.JavaConverters._

/** Contains methods for computing the Meteor score for pairs of sequences.
  *
  * @param  variablesCollections Graph collections in which to add the metric variables (for streaming metrics).
  * @param  valuesCollections    Graph collections in which to add the metric values.
  * @param  updatesCollections   Graph collections in which to add the metric updates.
  * @param  resetsCollections    Graph collections in which to add the metric resets.
  * @param  name                 Name for this metric.
  *
  * @author Emmanouil Antonios Platanios
  */
class Meteor protected (
    val normalize: Boolean = true,
    val ignoreCase: Boolean = true,
    val ignorePunctuation: Boolean = false,
    val variablesCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
    val valuesCollections: Set[Graph.Key[Output]] = Set(METRIC_VALUES),
    val updatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
    val resetsCollections: Set[Graph.Key[Op]] = Set(METRIC_RESETS),
    override val name: String = "Meteor"
)(implicit languages: Seq[(Language, Vocabulary)]) extends MTMetric {
  // TODO: Add support for more Meteor configuration options.
  // TODO: Do not ignore the weights.

  protected def getMeteorConfiguration(language: Language): MeteorConfiguration = {
    val meteorConfiguration = new MeteorConfiguration()
    meteorConfiguration.setLanguage(language.abbreviation)
    if (normalize && ignorePunctuation)
      meteorConfiguration.setNormalization(3)
    else if (normalize)
      meteorConfiguration.setNormalization(2)
    else if (ignoreCase)
      meteorConfiguration.setNormalization(1)
    else
      meteorConfiguration.setNormalization(0)
    meteorConfiguration
  }

  private[this] def statistics(batch: Seq[Tensor]): Tensor = {
    val (languageId, hyp, hypLen, ref, refLen) = (batch(0), batch(1), batch(2), batch(3), batch(4))
    val language = languageId.scalar.asInstanceOf[Int]

    val (hypSentences, hypLengths) = (hyp.unstack(), hypLen.unstack())
    val (refSentences, refLengths) = (ref.unstack(), refLen.unstack())
    val hypSeq = hypSentences.zip(hypLengths).map {
      case (s, len) =>
        val lenScalar = len.scalar.asInstanceOf[Int]
        val seq = s(0 :: lenScalar).entriesIterator.map(v => Encoding.tfStringToUTF8(v.asInstanceOf[String])).toSeq
        languages(language)._2.decodeSequence(seq)
    }
    val refSeq = refSentences.zip(refLengths).map {
      case (s, len) =>
        val lenScalar = len.scalar.asInstanceOf[Int]
        val seq = s(0 :: lenScalar).entriesIterator.map(v => Encoding.tfStringToUTF8(v.asInstanceOf[String])).toSeq
        languages(language)._2.decodeSequence(seq)
    }

    val meteorConfiguration = getMeteorConfiguration(languages(language)._1)
    val meteorScorer = new MeteorScorer(meteorConfiguration)
    val meteorStats = new MeteorStats()
    hypSeq.zip(refSeq).foreach(pair => {
      meteorStats.addStats(meteorScorer.getMeteorStats(
        new java.util.ArrayList[String](pair._1.asJava),
        new java.util.ArrayList[String](pair._2.asJava)))
    })

    Meteor.fromMeteorStats(meteorStats)
  }

  protected[Meteor] def score(statistics: Seq[Tensor]): Tensor = {
    val language = statistics(0).scalar.asInstanceOf[Int]
    val meteorStats = Meteor.toMeteorStats(statistics(1))
    val meteorConfiguration = getMeteorConfiguration(languages(language)._1)
    val meteorScorer = new MeteorScorer(meteorConfiguration)
    meteorScorer.computeMetrics(meteorStats)
    meteorStats.score.toFloat
  }

  override def compute(
      values: ((Output, Output, Output), (Output, Output)),
      weights: Output = null,
      name: String = this.name
  ): Output = {
    val ((languageId, src, srcLen), (tgt, tgtLen)) = values
    var ops = Set(src.op, srcLen.op, tgt.op, tgtLen.op)
    if (weights != null)
      ops += weights.op
    tf.createWithNameScope(name, ops) {
      val _statistics = tf.callback(
        statistics, Seq(languageId, src, srcLen, tgt, tgtLen), STRING, stateful = false, name = "Statistics")
      val _score = tf.callback(score, Seq(languageId, _statistics), FLOAT32, stateful = false, name = "Value")
      _score(0)
    }
  }

  override def streaming(
      values: ((Output, Output, Output), (Output, Output)),
      weights: Output,
      name: String = this.name
  ): Metric.StreamingInstance[Output] = {
    val ((languageId, src, srcLen), (tgt, tgtLen)) = values
    var ops = Set(src.op, srcLen.op, tgt.op, tgtLen.op)
    if (weights != null)
      ops += weights.op
    tf.createWithVariableScope(name) {
      tf.createWithNameScope(name, ops) {
        val statistics = variable(
          "Statistics", STRING, Shape(1), tf.ConstantInitializer(Tensor(new MeteorStats().toString())),
          variablesCollections)
        val _statistics = tf.callback(
          this.statistics, Seq(languageId, src, srcLen, tgt, tgtLen), STRING, stateful = false,
          name = "Statistics")
        val updateStatistics = statistics.assign(tf.callback(
          Meteor.aggregateStatistics, Seq(statistics.value, _statistics), STRING, stateful = false,
          name = "Statistics/Aggregate"))
        val value = tf.callback(score, Seq(languageId, _statistics), FLOAT32, stateful = false, name = "Value")
        val update = tf.callback(score, Seq(languageId, updateStatistics), FLOAT32, stateful = false, name = "Update")
        val reset = tf.group(Set(statistics.initializer), name = "Reset")
        valuesCollections.foreach(tf.currentGraph.addToCollection(value, _))
        updatesCollections.foreach(tf.currentGraph.addToCollection(update, _))
        resetsCollections.foreach(tf.currentGraph.addToCollection(reset, _))
        Metric.StreamingInstance(value, update, reset, Set(statistics))
      }
    }
  }
}

object Meteor {
  def apply(
      normalize: Boolean = true,
      ignoreCase: Boolean = true,
      ignorePunctuation: Boolean = false,
      variablesCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
      valuesCollections: Set[Graph.Key[Output]] = Set(METRIC_VALUES),
      updatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
      resetsCollections: Set[Graph.Key[Op]] = Set(METRIC_RESETS),
      name: String = "Meteor"
  )(implicit languages: Seq[(Language, Vocabulary)]): Meteor = {
    new Meteor(
      normalize, ignoreCase, ignorePunctuation, variablesCollections,
      valuesCollections, updatesCollections, resetsCollections, name)(languages)
  }

  protected[Meteor] def aggregateStatistics(statistics: Seq[Tensor]): Tensor = {
    val meteorStats1 = toMeteorStats(statistics(0))
    val meteorStats2 = toMeteorStats(statistics(1))
    meteorStats1.addStats(meteorStats2)
    fromMeteorStats(meteorStats1)
  }

  protected[Meteor] def toMeteorStats(statistics: Tensor): MeteorStats = {
    new MeteorStats(statistics.scalar.asInstanceOf[String])
  }

  protected[Meteor] def fromMeteorStats(meteorStats: MeteorStats): Tensor = {
    Tensor(meteorStats.toString)
  }
}
