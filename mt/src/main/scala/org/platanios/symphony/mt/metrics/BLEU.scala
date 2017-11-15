/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

import scala.collection.mutable

/** Contains methods for computing the
  * [BiLingual Evaluation Understudy (BLEU)](https://en.wikipedia.org/wiki/BLEU) score for pairs of sequences.
  *
  * @author Emmanouil Antonios Platanios
  */
object BLEU {
  /** BLEU score computation result.
    *
    * @param  score       Actual BLEU score.
    * @param  precisions  Precisions computed for each n-gram order.
    * @param  lengthRatio Ratio of hypothesis sequence lengths to reference sequence lengths. If multiple reference
    *                     sequences are provided, the minimum length of these sequences is used each time.
    * @param  smooth      Boolean value indicating whether the BLEU score was smoothed using the method described in
    *                     [Lin et al. 2004](https://dl.acm.org/citation.cfm?id=1219032).
    */
  case class Score(
      score: Double, precisions: Seq[Double], lengthRatio: Double, smooth: Boolean) {
    /** Maximum n-gram order used while computing the BLEU score. */
    val maxOrder: Int = precisions.size

    /** Brevity penalty computed for the BLEU score. */
    val brevityPenalty: Double = if (lengthRatio > 1) 1.0 else Math.exp(1.0 - 1.0 / lengthRatio)
  }

  /** Computes the BLEU score for the provided hypothesis sequences against one or more reference sequences.
    *
    * @param  referenceCorpus  Sequence of sequences of reference sequences to use as reference for each translation
    *                          sequence. Each reference and translation sequence is supposed to be a sequence over
    *                          tokens of type `T` (typically `String`).
    * @param  hypothesisCorpus Sequence of hypothesis sequences to score.
    * @param  maxOrder         Maximum n-gram order to use when computing the BLEU score.
    * @param  smooth           Boolean value indicating to use the smoothing method described in
    *                          [Lin et al. 2004](https://dl.acm.org/citation.cfm?id=1219032).
    * @return Computed BLEU score.
    */
  def computeBLEU[T](
      referenceCorpus: Seq[Seq[Seq[T]]],
      hypothesisCorpus: Seq[Seq[T]],
      maxOrder: Int = 4,
      smooth: Boolean = false): Score = {
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
    val bleu = geometricMean * brevityPenalty
    Score(bleu, precisions, lengthRatio, smooth)
  }
}
