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
    val smooth: Boolean = false,
    val variablesCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
    val valuesCollections: Set[Graph.Key[Output]] = Set(METRIC_VALUES),
    val updatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
    val resetsCollections: Set[Graph.Key[Op]] = Set(METRIC_RESETS),
    override val name: String = "BLEU"
)(implicit languages: Seq[(Language, Vocabulary)]) extends MTMetric {
  // TODO: Do not ignore the weights.

  private[this] def counts(batch: Seq[Tensor]): Seq[Tensor] = {
    val (tgtLanguageId, hyp, hypLen, ref, refLen) = (batch(0), batch(1), batch(2), batch(3), batch(4))
    val tgtLanguage = tgtLanguageId.scalar.asInstanceOf[Int]

    val (hypSentences, hypLengths) = (hyp.unstack(), hypLen.unstack())
    val (refSentences, refLengths) = (ref.unstack(), refLen.unstack())
    val hypSeq = hypSentences.zip(hypLengths).map {
      case (s, len) =>
        val lenScalar = len.scalar.asInstanceOf[Int]
        val seq = s(0 :: lenScalar).entriesIterator.map(v => Encoding.tfStringToUTF8(v.asInstanceOf[String])).toSeq
        languages(tgtLanguage)._2.decodeSequence(seq)
    }
    val refSeq = refSentences.zip(refLengths).map {
      case (s, len) =>
        val lenScalar = len.scalar.asInstanceOf[Int]
        val seq = s(0 :: lenScalar).entriesIterator.map(v => Encoding.tfStringToUTF8(v.asInstanceOf[String])).toSeq
        Seq(languages(tgtLanguage)._2.decodeSequence(seq))
    }
    val (matchesByOrder, possibleMatchesByOrder, _refLen, _hypLen) = BLEU.nGramMatches(refSeq, hypSeq, maxOrder)
    Seq(matchesByOrder, possibleMatchesByOrder, _refLen, _hypLen)
  }

  private[this] def score(
      matchesByOrder: Output,
      possibleMatchesByOrder: Output,
      referenceLength: Output,
      hypothesisLength: Output,
      name: String = "BLEU"
  ): Output = tf.createWithNameScope(name) {
    // Compute precisions
    val precisions = {
      if (smooth)
        (matchesByOrder.cast(FLOAT32) + 1.0f) / (possibleMatchesByOrder.cast(FLOAT32) + 1.0f)
      else
        tf.select(
          possibleMatchesByOrder > 0,
          matchesByOrder.cast(FLOAT32) / possibleMatchesByOrder.cast(FLOAT32),
          tf.zerosLike(matchesByOrder, FLOAT32))
    }

    // Compute the BLEU score
    val geometricMean = tf.cond(
      tf.min(precisions) > 0,
      () => tf.exp(tf.sum(tf.log(precisions) / maxOrder)),
      () => tf.zeros(FLOAT32, Shape()))

    val lengthRatio = hypothesisLength.cast(FLOAT32) / referenceLength.cast(FLOAT32)
    val brevityPenalty = tf.cond(
      lengthRatio > 1,
      () => tf.ones(FLOAT32, Shape()),
      () => tf.exp(1.0f - (1.0f / lengthRatio)))

    geometricMean * brevityPenalty * 100
  }

  override def compute(
      values: ((Output, Output, Output), (Output, Output)),
      weights: Output = null,
      name: String = this.name
  ): Output = {
    val ((tgtLanguageId, src, srcLen), (tgt, tgtLen)) = values
    var ops = Set(src.op, srcLen.op, tgt.op, tgtLen.op)
    if (weights != null)
      ops += weights.op
    tf.createWithNameScope(name, ops) {
      val _counts = tf.callback(
        counts, Seq(tgtLanguageId, src, srcLen, tgt, tgtLen), Seq(INT64, INT64, INT32, INT32), stateful = false)
      val (_matches, _possibleMatches, _refLen, _hypLen) = (_counts(0), _counts(1), _counts(2), _counts(3))
      score(_matches, _possibleMatches, _refLen, _hypLen, name = "Value")
    }
  }

  override def streaming(
      values: ((Output, Output, Output), (Output, Output)),
      weights: Output,
      name: String = this.name
  ): Metric.StreamingInstance[Output] = {
    val ((tgtLanguageId, src, srcLen), (tgt, tgtLen)) = values
    var ops = Set(src.op, srcLen.op, tgt.op, tgtLen.op)
    if (weights != null)
      ops += weights.op
    tf.variableScope(name) {
      tf.createWithNameScope(name, ops) {
        val n = maxOrder
        val matches = variable("Matches", INT64, Shape(n), tf.ZerosInitializer, variablesCollections)
        val possibleMatches = variable("PossibleMatches", INT64, Shape(n), tf.ZerosInitializer, variablesCollections)
        val refLen = variable("ReferenceLength", INT32, Shape(), tf.ZerosInitializer, variablesCollections)
        val hypLen = variable("HypothesisLength", INT32, Shape(), tf.ZerosInitializer, variablesCollections)
        val _counts = tf.callback(
          counts, Seq(tgtLanguageId, src, srcLen, tgt, tgtLen), Seq(INT64, INT64, INT32, INT32), stateful = false)
        val (_matches, _possibleMatches, _refLen, _hypLen) = (_counts(0), _counts(1), _counts(2), _counts(3))
        val updateMatches = matches.assignAdd(_matches)
        val updatePossibleMatches = possibleMatches.assignAdd(_possibleMatches)
        val updateRefLen = refLen.assignAdd(_refLen)
        val updateHypLen = hypLen.assignAdd(_hypLen)
        val value = score(matches, possibleMatches, refLen, hypLen, name = "Value")
        val update = score(updateMatches, updatePossibleMatches, updateRefLen, updateHypLen, name = "Update")
        val reset = tf.group(
          Set(matches.initializer, possibleMatches.initializer, refLen.initializer, hypLen.initializer),
          name = "Reset")
        valuesCollections.foreach(tf.currentGraph.addToCollection(value, _))
        updatesCollections.foreach(tf.currentGraph.addToCollection(update, _))
        resetsCollections.foreach(tf.currentGraph.addToCollection(reset, _))
        Metric.StreamingInstance(value, update, reset, Set(matches, possibleMatches, refLen, hypLen))
      }
    }
  }
}

object BLEU {
  def apply(
      maxOrder: Int = 4,
      smooth: Boolean = false,
      variablesCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
      valuesCollections: Set[Graph.Key[Output]] = Set(METRIC_VALUES),
      updatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
      resetsCollections: Set[Graph.Key[Op]] = Set(METRIC_RESETS),
      name: String = "BLEU"
  )(implicit languages: Seq[(Language, Vocabulary)]): BLEU = {
    new BLEU(
      maxOrder, smooth, variablesCollections,
      valuesCollections, updatesCollections, resetsCollections, name)(languages)
  }

  /** BLEU score computation result.
    *
    * @param  score       Actual BLEU score.
    * @param  precisions  Precisions computed for each n-gram order.
    * @param  lengthRatio Ratio of hypothesis sequence lengths to reference sequence lengths. If multiple reference
    *                     sequences are provided, the minimum length of these sequences is used each time.
    * @param  smooth      Boolean value indicating whether the BLEU score was smoothed using the method described in
    *                     [Lin et al. 2004](https://dl.acm.org/citation.cfm?id=1219032).
    */
  case class BLEUScore(
      score: Double, precisions: Seq[Double], lengthRatio: Double, smooth: Boolean) {
    /** Maximum n-gram order used while computing the BLEU score. */
    val maxOrder: Int = precisions.size

    /** Brevity penalty computed for the BLEU score. */
    val brevityPenalty: Double = if (lengthRatio > 1) 1.0 else Math.exp(1.0 - 1.0 / lengthRatio)
  }

  /** Computes the BLEU score for the provided hypothesis sequences against one or more reference sequences.
    *
    * @param  referenceCorpus  Sequence of sequences of reference sequences to use as reference for each translation
    *                          sequence. Each reference and hypothesis sequence is supposed to be a sequence over
    *                          tokens of type `T` (typically `String`).
    * @param  hypothesisCorpus Sequence of hypothesis sequences to score.
    * @param  maxOrder         Maximum n-gram order to use when computing the BLEU score.
    * @param  smooth           Boolean value indicating to use the smoothing method described in
    *                          [Lin et al. 2004](https://dl.acm.org/citation.cfm?id=1219032).
    * @return Computed BLEU score.
    */
  def bleu[T](
      referenceCorpus: Seq[Seq[Seq[T]]],
      hypothesisCorpus: Seq[Seq[T]],
      maxOrder: Int = 4,
      smooth: Boolean = false): BLEUScore = {
    // Compute counts for matches and possible matches
    val (matchesByOrder, possibleMatchesByOrder, referenceLength, hypothesisLength) =
      nGramMatches(referenceCorpus, hypothesisCorpus, maxOrder)

    // Compute precisions
    val precisions = matchesByOrder.zip(possibleMatchesByOrder).map {
      case (matches, possibleMatches) =>
        if (smooth) {
          (matches + 1.0) / (possibleMatches + 1.0)
        } else {
          if (possibleMatches > 0)
            matches.toDouble / possibleMatches
          else
            0.0
        }
    }

    // Compute the BLEU score
    val geometricMean = if (precisions.min > 0) Math.exp(precisions.map(Math.log(_) / maxOrder).sum) else 0.0
    val lengthRatio = hypothesisLength.toDouble / referenceLength.toDouble
    val brevityPenalty = if (lengthRatio > 1) 1.0 else Math.exp(1.0 - 1.0 / lengthRatio)
    val bleu = geometricMean * brevityPenalty * 100
    BLEUScore(bleu, precisions, lengthRatio, smooth)
  }

  private[evaluation] def nGramMatches[T](
      referenceCorpus: Seq[Seq[Seq[T]]],
      hypothesisCorpus: Seq[Seq[T]],
      maxOrder: Int = 4
  ): (Array[Long], Array[Long], Int, Int) = {
    // Compute counts for matches and possible matches
    val matchesByOrder = mutable.ArrayBuffer.fill(maxOrder)(0L)
    val possibleMatchesByOrder = mutable.ArrayBuffer.fill(maxOrder)(0L)
    var referenceLength: Int = 0
    var hypothesisLength: Int = 0
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
