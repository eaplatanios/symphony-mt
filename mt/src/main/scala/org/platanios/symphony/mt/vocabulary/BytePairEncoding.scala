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

package org.platanios.symphony.mt.vocabulary

import scala.collection.mutable

// TODO: !!! This is incomplete.

/** Use byte pair encoding (BPE) to learn a variable-length encoding of the vocabulary in a text.
  * Unlike the original BPE, it does not compress the plain text, but can be used to reduce the vocabulary of a text to
  * a configurable number of symbols, with only a small increase in the number of tokens.
  *
  * '''Reference:'''
  *
  * Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
  * Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
  *
  * @param  wordCounts
  * @param  numSymbols
  * @param  minFreq
  *
  * @author Emmanouil Antonios Platanios
  */
class BytePairEncoding(
    val wordCounts: Iterable[(Long, String)],
    val numSymbols: Int = 10000,
    val minFreq: Int = 2
) {
  protected val sortedWordCounts: Seq[(Long, String)] = wordCounts.toSeq.sortBy(-_._1)

  protected def pairCounts(wordCounts: Seq[(Long, String)]): BytePairEncoding.PairStatistics = {
    var counts: Map[String, Long] = Map.empty.withDefaultValue(0)
    var indices: Map[(String, Int), Long] = Map.empty.withDefaultValue(0)
    wordCounts.zipWithIndex.foreach {
      case ((count, word), index) =>
        word.sliding(2).foreach(pair => {
          counts = counts.updated(pair, counts(pair) + count)
          indices = indices.updated((pair, index), indices((pair, index)) + 1)
        })
    }
    BytePairEncoding.PairStatistics(counts, indices)
  }
}

object BytePairEncoding {
  case class PairStatistics(counts: Map[String, Long], indices: Map[(String, Int), Long])

  /** Prunes the symbol counts map for improved efficiency.
    *
    * The frequency of a symbol pair never increases, so pruning is generally safe (until the most frequent pair is less
    * frequent than a pair that was previously pruned).
    *
    * @param  counts     Symbol counts map to be pruned.
    * @param  fullCounts Map that keeps counts information for when we need to access pruned symbol counts. Note that
    *                    this map may be modified by this method.
    * @param  threshold  Symbol count threshold to use while pruning.
    * @return Pruned symbol counts map.
    */
  protected def pruneCounts(
      counts: Map[String, Long],
      fullCounts: mutable.Map[String, Long],
      threshold: Long
  ): Map[String, Long] = {
    counts.filter {
      case (symbol, count) =>
        if (count < 0)
          fullCounts.update(symbol, fullCounts(symbol) + count)
        else
          fullCounts.update(symbol, count)
        count < threshold
    }
  }
}
