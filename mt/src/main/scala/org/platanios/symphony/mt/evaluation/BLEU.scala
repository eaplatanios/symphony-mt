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

import scala.collection.mutable

// TODO: Use weights.

/** Contains methods for computing the
  * [BiLingual Evaluation Understudy (BLEU)](https://en.wikipedia.org/wiki/BLEU) score for pairs of sequences.
  *
  * @param  variablesCollections Graph collections in which to add the metric variables (for streaming metrics).
  * @param  valuesCollections    Graph collections in which to add the metric values.
  * @param  updatesCollections   Graph collections in which to add the metric updates.
  * @param  resetsCollections    Graph collections in which to add the metric resets.
  * @param  name                 Name for this metric.
  *
  * @author Emmanouil Antonios Platanios
  */
class BLEU protected (
    val maxOrder: Int = 4,
    val smoothing: BLEU.Smoothing = BLEU.NoSmoothing,
    val variablesCollections: Set[Graph.Key[Variable[Any]]] = Set(METRIC_VARIABLES),
    val valuesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_VALUES),
    val updatesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_UPDATES),
    val resetsCollections: Set[Graph.Key[UntypedOp]] = Set(METRIC_RESETS),
    override val name: String = "BLEU"
)(implicit languages: Seq[(Language, Vocabulary)]) extends MTMetric {
  // TODO: Do not ignore the weights.

  private def counts(
      tgtLanguageId: Tensor[Int],
      hypotheses: Tensor[String],
      hypothesisLengths: Tensor[Int],
      references: Tensor[String],
      referenceLengths: Tensor[Int]
  ): (Tensor[Long], Tensor[Long], Tensor[Long], Tensor[Long]) = {
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
        Seq(languages(tgtLanguage)._2.decodeSequence(seq))
    }
    val (matchesByOrder, possibleMatchesByOrder, refLen, hypLen) = BLEU.nGramMatches(refSeq, hypSeq, maxOrder)
    (matchesByOrder, possibleMatchesByOrder, refLen, hypLen)
  }

  private def score(
      matchesByOrder: Output[Long],
      possibleMatchesByOrder: Output[Long],
      referenceLengths: Output[Long],
      hypothesisLengths: Output[Long],
      name: String = "BLEU"
  ): Output[Float] = {
    tf.nameScope(name) {
      // Compute precisions.
      val precisions = tf.select(
        possibleMatchesByOrder > 0L,
        tf.log(smoothing(matchesByOrder, possibleMatchesByOrder, referenceLengths, hypothesisLengths)),
        tf.zerosLike(matchesByOrder).toFloat)

      // Compute the BLEU score.
      val geometricMean = tf.exp(tf.sum(precisions) / maxOrder)
      val lengthRatio = referenceLengths.toFloat / hypothesisLengths.toFloat
      val brevityPenalty = tf.minimum(1.0f, tf.exp(1.0f - lengthRatio))
      geometricMean * brevityPenalty * 100
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
      val _counts = tf.callback(
        function = Function.tupled(counts _),
        input = (tgtLanguageId, hypSentences, hypSentenceLengths, refSentences, refSentenceLengths),
        outputDataType = (INT64, INT64, INT64, INT64),
        stateful = false,
        name = "Counts")
      score(_counts._1, _counts._2, _counts._3, _counts._4, name = "Value")
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
        val n = maxOrder
        val matches = variable[Long]("Matches", Shape(n), tf.ZerosInitializer, variablesCollections)
        val possibleMatches = variable[Long]("PossibleMatches", Shape(n), tf.ZerosInitializer, variablesCollections)
        val refLen = variable[Long]("ReferenceLength", Shape(), tf.ZerosInitializer, variablesCollections)
        val hypLen = variable[Long]("HypothesisLength", Shape(), tf.ZerosInitializer, variablesCollections)
        val _counts = tf.callback(
          function = Function.tupled(counts _),
          input = (tgtLanguageId, hypSentences, hypSentenceLengths, refSentences, refSentenceLengths),
          outputDataType = (INT64, INT64, INT64, INT64),
          stateful = false,
          name = "Counts")
        val updateMatches = matches.assignAdd(_counts._1)
        val updatePossibleMatches = possibleMatches.assignAdd(_counts._2)
        val updateRefLen = refLen.assignAdd(_counts._3)
        val updateHypLen = hypLen.assignAdd(_counts._4)
        val value = score(matches, possibleMatches, refLen, hypLen, name = "Value")
        val update = score(updateMatches, updatePossibleMatches, updateRefLen, updateHypLen, name = "Update")
        val reset = tf.group(
          Set(matches.initializer, possibleMatches.initializer, refLen.initializer, hypLen.initializer),
          name = "Reset")
        valuesCollections.foreach(tf.currentGraph.addToCollection(_)(value.asUntyped))
        updatesCollections.foreach(tf.currentGraph.addToCollection(_)(update.asUntyped))
        resetsCollections.foreach(tf.currentGraph.addToCollection(_)(reset.asUntyped))
        Metric.StreamingInstance(
          value, update, reset,
          Set(matches.asUntyped, possibleMatches.asUntyped, refLen.asUntyped, hypLen.asUntyped))
      }
    }
  }
}

object BLEU {
  def apply(
      maxOrder: Int = 4,
      smoothing: Smoothing = NoSmoothing,
      variablesCollections: Set[Graph.Key[Variable[Any]]] = Set(METRIC_VARIABLES),
      valuesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_VALUES),
      updatesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_UPDATES),
      resetsCollections: Set[Graph.Key[UntypedOp]] = Set(METRIC_RESETS),
      name: String = "BLEU"
  )(implicit languages: Seq[(Language, Vocabulary)]): BLEU = {
    new BLEU(
      maxOrder, smoothing, variablesCollections,
      valuesCollections, updatesCollections, resetsCollections, name
    )(languages)
  }

  case class Counts(
      matchesByOrder: Tensor[Long],
      possibleMatchesByOrder: Tensor[Long],
      referenceLength: Tensor[Long],
      hypothesisLength: Tensor[Long])

  trait Smoothing {
    def apply(
        matchesByOrder: Output[Long],
        possibleMatchesByOrder: Output[Long],
        referenceLengths: Output[Long],
        hypothesisLengths: Output[Long]
    ): Output[Float]
  }

  case object NoSmoothing extends Smoothing {
    override def apply(
        matchesByOrder: Output[Long],
        possibleMatchesByOrder: Output[Long],
        referenceLengths: Output[Long],
        hypothesisLengths: Output[Long]
    ): Output[Float] = {
      matchesByOrder.toFloat / possibleMatchesByOrder.toFloat
    }
  }

  case class EpsilonSmoothing(epsilon: Float) extends Smoothing {
    override def apply(
        matchesByOrder: Output[Long],
        possibleMatchesByOrder: Output[Long],
        referenceLengths: Output[Long],
        hypothesisLengths: Output[Long]
    ): Output[Float] = {
      tf.select(
        tf.equal(matchesByOrder, 0L),
        tf.fill[Float, Long](tf.shape(matchesByOrder))(epsilon) / possibleMatchesByOrder.toFloat,
        matchesByOrder.toFloat / possibleMatchesByOrder.toFloat)
    }
  }

  case object LinOchSmoothing extends Smoothing {
    override def apply(
        matchesByOrder: Output[Long],
        possibleMatchesByOrder: Output[Long],
        referenceLengths: Output[Long],
        hypothesisLengths: Output[Long]
    ): Output[Float] = {
      val ones = tf.concatenate(Seq(
        tf.zeros[Float](Shape(1)),
        tf.ones[Float](tf.shape(matchesByOrder).slice(0) - 1)
      ))
      tf.add(matchesByOrder.toFloat, ones) / tf.add(possibleMatchesByOrder.toFloat, ones)
    }
  }

  case object NISTSmoothing extends Smoothing {
    override def apply(
        matchesByOrder: Output[Long],
        possibleMatchesByOrder: Output[Long],
        referenceLengths: Output[Long],
        hypothesisLengths: Output[Long]
    ): Output[Float] = {
      val zeros = tf.equal(matchesByOrder, 0L).toFloat
      val factors = 1.0f / tf.pow(2.0f, tf.cumsum(zeros, axis = 0))
      tf.select(
        tf.equal(matchesByOrder, 0L),
        factors,
        matchesByOrder.toFloat / possibleMatchesByOrder.toFloat)
    }
  }

  private[evaluation] def nGramMatches[T](
      referenceCorpus: Seq[Seq[Seq[T]]],
      hypothesisCorpus: Seq[Seq[T]],
      maxOrder: Int = 4
  ): (Array[Long], Array[Long], Long, Long) = {
    // Compute counts for matches and possible matches
    val matchesByOrder = mutable.ArrayBuffer.fill(maxOrder)(0L)
    val possibleMatchesByOrder = mutable.ArrayBuffer.fill(maxOrder)(0L)
    var referenceLength: Long = 0L
    var hypothesisLength: Long = 0L
    referenceCorpus.zip(hypothesisCorpus).foreach {
      case (references, hypothesis) =>
        referenceLength += references.map(_.size).min
        hypothesisLength += hypothesis.size

        // Compute n-gram counts
        val refNGramCounts = references.map(Utilities.countNGrams(_, maxOrder))
        val hypNGramCounts = Utilities.countNGrams(hypothesis, maxOrder)
        val overlapNGramCounts = hypNGramCounts.map {
          case (ngram, count) => ngram -> Math.min(refNGramCounts.map(_.getOrElse(ngram, 0L)).max, count)
        }

        // Update counts for matches and possible matches
        overlapNGramCounts.foreach(p => matchesByOrder(p._1.size - 1) += p._2)
        (0 until maxOrder).foreach(n => {
          val possibleMatches = hypothesis.size - n
          if (possibleMatches > 0)
            possibleMatchesByOrder(n) += possibleMatches
        })
    }
    (matchesByOrder.toArray, possibleMatchesByOrder.toArray, referenceLength, hypothesisLength)
  }
}
