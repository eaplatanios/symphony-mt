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
    val smoothing: BLEU.Smoothing = BLEU.NoSmoothing,
    val variablesCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
    val valuesCollections: Set[Graph.Key[Output]] = Set(METRIC_VALUES),
    val updatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
    val resetsCollections: Set[Graph.Key[Op]] = Set(METRIC_RESETS),
    override val name: String = "BLEU"
)(implicit languages: Seq[(Language, Vocabulary)]) extends MTMetric {
  // TODO: Do not ignore the weights.

  private[this] def counts(batch: Seq[Tensor[DataType]]): Seq[Tensor[DataType]] = {
    val (tgtLanguageId, hyp, hypLen, ref, refLen) = (batch(0), batch(1), batch(2), batch(3), batch(4))
    val tgtLanguage = tgtLanguageId.scalar.asInstanceOf[Long]

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
    Seq[Tensor[DataType]](matchesByOrder.toTensor, possibleMatchesByOrder.toTensor, _refLen, _hypLen)
  }

  private[this] def score(
      matchesByOrder: Output,
      possibleMatchesByOrder: Output,
      referenceLength: Output,
      hypothesisLength: Output,
      name: String = "BLEU"
  ): Output = tf.createWithNameScope(name) {
    // Compute precisions.
    val precisions = tf.select(
      possibleMatchesByOrder > 0,
      tf.log(smoothing(matchesByOrder, possibleMatchesByOrder, referenceLength, hypothesisLength)),
      tf.zerosLike(matchesByOrder, FLOAT32))

    // Compute the BLEU score.
    val geometricMean = tf.exp(tf.sum(precisions) / maxOrder)
    val lengthRatio = referenceLength.cast(FLOAT32) / hypothesisLength.cast(FLOAT32)
    val brevityPenalty = tf.minimum(1.0f, tf.exp(1.0f - lengthRatio))
    geometricMean * brevityPenalty * 100
  }

  override def compute(
      values: ((Output, Output, Output), (Output, Output)),
      weights: Option[Output] = None,
      name: String = this.name
  ): Output = {
    val ((tgtLanguageId, src, srcLen), (tgt, tgtLen)) = values
    var ops = Set(src.op, srcLen.op, tgt.op, tgtLen.op)
    weights.foreach(ops += _.op)
    tf.createWithNameScope(name, ops) {
      val _counts = tf.callback(
        counts, Seq(tgtLanguageId, src, srcLen, tgt, tgtLen), Seq[DataType](INT64, INT64, INT32, INT32),
        stateful = false)
      val (_matches, _possibleMatches, _refLen, _hypLen) = (_counts(0), _counts(1), _counts(2), _counts(3))
      score(_matches, _possibleMatches, _refLen, _hypLen, name = "Value")
    }
  }

  override def streaming(
      values: ((Output, Output, Output), (Output, Output)),
      weights: Option[Output] = None,
      name: String = this.name
  ): Metric.StreamingInstance[Output] = {
    val ((tgtLanguageId, src, srcLen), (tgt, tgtLen)) = values
    var ops = Set(src.op, srcLen.op, tgt.op, tgtLen.op)
    weights.foreach(ops += _.op)
    tf.variableScope(name) {
      tf.createWithNameScope(name, ops) {
        val n = maxOrder
        val matches = variable("Matches", INT64, Shape(n), tf.ZerosInitializer, variablesCollections)
        val possibleMatches = variable("PossibleMatches", INT64, Shape(n), tf.ZerosInitializer, variablesCollections)
        val refLen = variable("ReferenceLength", INT32, Shape(), tf.ZerosInitializer, variablesCollections)
        val hypLen = variable("HypothesisLength", INT32, Shape(), tf.ZerosInitializer, variablesCollections)
        val _counts = tf.callback(
          counts, Seq(tgtLanguageId, src, srcLen, tgt, tgtLen), Seq[DataType](INT64, INT64, INT32, INT32),
          stateful = false)
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
      smoothing: Smoothing = NoSmoothing,
      variablesCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
      valuesCollections: Set[Graph.Key[Output]] = Set(METRIC_VALUES),
      updatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
      resetsCollections: Set[Graph.Key[Op]] = Set(METRIC_RESETS),
      name: String = "BLEU"
  )(implicit languages: Seq[(Language, Vocabulary)]): BLEU = {
    new BLEU(
      maxOrder, smoothing, variablesCollections,
      valuesCollections, updatesCollections, resetsCollections, name)(languages)
  }

  trait Smoothing {
    def apply(
        matchesByOrder: Output,
        possibleMatchesByOrder: Output,
        referenceLength: Output,
        hypothesisLength: Output
    ): Output
  }

  case object NoSmoothing extends Smoothing {
    override def apply(
        matchesByOrder: Output,
        possibleMatchesByOrder: Output,
        referenceLength: Output,
        hypothesisLength: Output
    ): Output = {
      matchesByOrder.cast(FLOAT32) / possibleMatchesByOrder.cast(FLOAT32)
    }
  }

  case class EpsilonSmoothing(epsilon: Float) extends Smoothing {
    override def apply(
        matchesByOrder: Output,
        possibleMatchesByOrder: Output,
        referenceLength: Output,
        hypothesisLength: Output
    ): Output = {
      tf.select(
        tf.equal(matchesByOrder, 0),
        tf.fill(FLOAT32, tf.shape(matchesByOrder))(epsilon) / possibleMatchesByOrder.cast(FLOAT32),
        matchesByOrder.cast(FLOAT32) / possibleMatchesByOrder.cast(FLOAT32))
    }
  }

  case object LinOchSmoothing extends Smoothing {
    override def apply(
        matchesByOrder: Output,
        possibleMatchesByOrder: Output,
        referenceLength: Output,
        hypothesisLength: Output
    ): Output = {
      val ones = tf.concatenate(Seq(tf.zeros(FLOAT32, Shape(1)), tf.ones(FLOAT32, tf.shape(matchesByOrder)(0) - 1)))
      (matchesByOrder.cast(FLOAT32) + ones) / (possibleMatchesByOrder.cast(FLOAT32) + ones)
    }
  }

  case object NISTSmoothing extends Smoothing {
    override def apply(
        matchesByOrder: Output,
        possibleMatchesByOrder: Output,
        referenceLength: Output,
        hypothesisLength: Output
    ): Output = {
      val zeros = tf.equal(matchesByOrder, 0).cast(FLOAT32)
      val factors = 1.0f / tf.pow(2.0f, tf.cumsum(zeros))
      tf.select(
        tf.equal(matchesByOrder, 0),
        factors * matchesByOrder.cast(FLOAT32) / possibleMatchesByOrder.cast(FLOAT32),
        matchesByOrder.cast(FLOAT32) / possibleMatchesByOrder.cast(FLOAT32))
    }
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
