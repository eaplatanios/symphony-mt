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
import org.platanios.symphony.mt.utilities.Encoding
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.ops.metrics.Metric._

import edu.cmu.meteor.scorer.MeteorConfiguration
import edu.cmu.meteor.scorer.MeteorScorer
import edu.cmu.meteor.scorer.MeteorStats

import scala.collection.JavaConverters._
import scala.collection.concurrent

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
    val variablesCollections: Set[Graph.Key[Variable[Any]]] = Set(METRIC_VARIABLES),
    val valuesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_VALUES),
    val updatesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_UPDATES),
    val resetsCollections: Set[Graph.Key[UntypedOp]] = Set(METRIC_RESETS),
    override val name: String = "Meteor"
)(implicit languages: Seq[(Language, Vocabulary)]) extends MTMetric {
  protected val meteorScorerCache: concurrent.Map[Language, Option[MeteorScorer]] = concurrent.TrieMap.empty

  // TODO: Add support for more Meteor configuration options.
  // TODO: Do not ignore the weights.

  protected def getMeteorScorer(language: Language): Option[MeteorScorer] = {
    meteorScorerCache.getOrElseUpdate(language, {
      // TODO: Find a better way to deal with unsupported languages.
      try {
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
        Some(new MeteorScorer(meteorConfiguration))
      } catch {
        case _: Throwable => None
      }
    })
  }

  private def statistics(
      tgtLanguageId: Tensor[Int],
      hypotheses: Tensor[String],
      hypothesisLengths: Tensor[Int],
      references: Tensor[String],
      referenceLengths: Tensor[Int]
  ): Tensor[String] = {
    val tgtLanguage = tgtLanguageId.scalar
    val (hypSentences, hypLengths) = (hypotheses.unstack(), hypothesisLengths.unstack())
    val (refSentences, refLengths) = (references.unstack(), referenceLengths.unstack())
    val hypSeq = hypSentences.zip(hypLengths).map {
      case (s, len) =>
        val lenScalar = len.scalar
        val seq = s(0 :: lenScalar).entriesIterator.map(v => Encoding.tfStringToUTF8(v)).toSeq
        languages(tgtLanguage)._2.decodeSequence(seq)
    }
    val refSeq = refSentences.zip(refLengths).map {
      case (s, len) =>
        val lenScalar = len.scalar
        val seq = s(0 :: lenScalar).entriesIterator.map(v => Encoding.tfStringToUTF8(v)).toSeq
        languages(tgtLanguage)._2.decodeSequence(seq)
    }

    getMeteorScorer(languages(tgtLanguage)._1) match {
      case Some(meteorScorer) =>
        val meteorStats = new MeteorStats()
        hypSeq.zip(refSeq).foreach(pair => {
          meteorStats.addStats(meteorScorer.getMeteorStats(
            new java.util.ArrayList[String](pair._1.asJava),
            new java.util.ArrayList[String](pair._2.asJava)))
        })

        Meteor.fromMeteorStats(meteorStats)
      case None => Tensor("")
    }
  }

  protected def score(languageId: Tensor[Int], statistics: Tensor[String]): Tensor[Float] = {
    val language = languageId.scalar
    Meteor.toMeteorStats(statistics) match {
      case Some(stats) => getMeteorScorer(languages(language)._1) match {
        case Some(meteorScorer) =>
          meteorScorer.computeMetrics(stats)
          100 * stats.score.toFloat
        case None => Float.NaN
      }
      case None => Float.NaN
    }
  }

  override def compute(
      values: (SentencesWithLanguage[String], (SentencesWithLanguagePair[String], Sentences[String])),
      weights: Option[Output[Float]] = None,
      name: String = this.name
  ): Output[Float] = {
    val tgtLanguageId = values._1._1
    val hypSentences = values._1._2._1
    val hypSentenceLengths = values._1._2._2
    val refSentences = values._2._2._1
    val refSentenceLengths = values._2._2._2
    tf.nameScope(name) {
      val _statistics = tf.callback(
        function = Function.tupled(statistics _),
        input = (tgtLanguageId, hypSentences, hypSentenceLengths, refSentences, refSentenceLengths),
        outputDataType = STRING,
        stateful = false,
        name = "Statistics")
      tf.callback(
        function = Function.tupled(score _),
        input = (tgtLanguageId, _statistics),
        outputDataType = FLOAT32,
        stateful = false,
        name = "Value")
    }
  }

  override def streaming(
      values: (SentencesWithLanguage[String], (SentencesWithLanguagePair[String], Sentences[String])),
      weights: Option[Output[Float]] = None,
      name: String = this.name
  ): Metric.StreamingInstance[Output[Float]] = {
    val tgtLanguageId = values._1._1
    val hypSentences = values._1._2._1
    val hypSentenceLengths = values._1._2._2
    val refSentences = values._2._2._1
    val refSentenceLengths = values._2._2._2
    tf.variableScope(name) {
      tf.nameScope(name) {
        // TODO: Find a better way to deal with the language.
        val language = variable[Int]("Language", Shape(), tf.ZerosInitializer, variablesCollections)
        val statistics = variable[String]("Statistics", Shape(1), tf.ConstantInitializer(Tensor(new MeteorStats().toString())), variablesCollections)
        val _statistics = tf.callback(
          function = Function.tupled(this.statistics _),
          input = (tgtLanguageId, hypSentences, hypSentenceLengths, refSentences, refSentenceLengths),
          outputDataType = STRING,
          stateful = false,
          name = "Statistics")
        val updateLanguage = language.assign(tgtLanguageId, "Language/Update")
        val updateStatistics = statistics.assign(tf.callback(
          function = Function.tupled(Meteor.aggregateStatistics _),
          input = (statistics.value, _statistics),
          outputDataType = STRING,
          stateful = false,
          name = "Statistics/Update"))
        val value = tf.callback(
          function = Function.tupled(score _),
          input = (language.value, statistics.value),
          outputDataType = FLOAT32,
          stateful = false,
          name = "Value")
        val update = tf.callback(
          function = Function.tupled(score _),
          input = (updateLanguage, updateStatistics),
          outputDataType = FLOAT32,
          stateful = false,
          name = "Update")
        val reset = tf.group(Set(statistics.initializer), name = "Reset")
        valuesCollections.foreach(tf.currentGraph.addToCollection(_)(value.asUntyped))
        updatesCollections.foreach(tf.currentGraph.addToCollection(_)(update.asUntyped))
        resetsCollections.foreach(tf.currentGraph.addToCollection(_)(reset.asUntyped))
        Metric.StreamingInstance(value, update, reset, Set(language.asUntyped, statistics.asUntyped))
      }
    }
  }
}

object Meteor {
  def apply(
      normalize: Boolean = true,
      ignoreCase: Boolean = true,
      ignorePunctuation: Boolean = false,
      variablesCollections: Set[Graph.Key[Variable[Any]]] = Set(METRIC_VARIABLES),
      valuesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_VALUES),
      updatesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_UPDATES),
      resetsCollections: Set[Graph.Key[UntypedOp]] = Set(METRIC_RESETS),
      name: String = "Meteor"
  )(implicit languages: Seq[(Language, Vocabulary)]): Meteor = {
    new Meteor(
      normalize, ignoreCase, ignorePunctuation, variablesCollections,
      valuesCollections, updatesCollections, resetsCollections, name)(languages)
  }

  protected[Meteor] def aggregateStatistics(
      statistics1: Tensor[String],
      statistics2: Tensor[String]
  ): Tensor[String] = {
    val meteorStats1 = toMeteorStats(statistics1)
    val meteorStats2 = toMeteorStats(statistics2)
    (meteorStats1, meteorStats2) match {
      case (Some(stats1), Some(stats2)) =>
        stats1.addStats(stats2)
        fromMeteorStats(stats1)
      case _ => Tensor("")
    }
  }

  protected[Meteor] def toMeteorStats(
      statistics: Tensor[String]
  ): Option[MeteorStats] = {
    val s = statistics.scalar
    if (s != "")
      Some(new MeteorStats(s))
    else
      None
  }

  protected[Meteor] def fromMeteorStats(meteorStats: MeteorStats): Tensor[String] = {
    Tensor(meteorStats.toString)
  }
}
