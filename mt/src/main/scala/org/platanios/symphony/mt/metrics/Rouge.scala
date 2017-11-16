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

/** Contains methods for computing 
  * [Recall-Oriented Understudy for Gisting Evaluation (ROUGE)](https://en.wikipedia.org/wiki/ROUGE_(metric)) scores 
  * for pairs of sequences.
  *
  * @author Emmanouil Antonios Platanios
  */
object Rouge {
  case class RougeNScore(n: Int, precision: Double, recall: Double, f1Score: Double)

  /** Computes the Rouge-N score for the provided hypothesis sequences against the provided reference sequences.
    *
    * @param  referenceCorpus  Sequence of reference sequences. Each reference and hypothesis sequence is supposed to
    *                          be a sequence over tokens of type `T` (typically `String`).
    * @param  hypothesisCorpus Sequence of hypothesis sequences.
    * @param  n                n-gram order to use.
    * @return Computed Rouge-N score.
    */
  def rougeN[T](referenceCorpus: Seq[Seq[T]], hypothesisCorpus: Seq[Seq[T]], n: Int = 2): RougeNScore = {
    val scores = referenceCorpus.zip(hypothesisCorpus).map {
      case (reference, hypothesis) =>
        // Compute the reference and the hypothesis n-grams
        val referenceNGrams = reference.iterator.sliding(n).withPartial(false).toSet
        val hypothesisNGrams = hypothesis.iterator.sliding(n).withPartial(false).toSet

        // Compute the overlapping n-grams between the reference and the hypothesis sequences
        val overlappingNGrams = referenceNGrams.intersect(hypothesisNGrams)

        // Handle the edge case in a way that is not mathematically correct, but is good enough
        val precision = if (hypothesisNGrams.isEmpty) 0.0 else overlappingNGrams.size / hypothesisNGrams.size
        val recall = if (referenceNGrams.isEmpty) 0.0 else overlappingNGrams.size / referenceNGrams.size
        val f1Score = 2.0 * precision * recall / (precision + recall + 1e-8)
        (precision, recall, f1Score)
    }
    val scoresSum = scores.reduce[(Double, Double, Double)]((s1, s2) => (s1._1 + s2._1, s1._2 + s2._2, s1._3 + s2._3))
    RougeNScore(n, scoresSum._1 / scores.size, scoresSum._2 / scores.size, scoresSum._3 / scores.size)
  }
}
