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

package org.platanios.symphony.mt.data.scores

import org.platanios.symphony.mt.Language

import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
class SentenceRarity(
    val wordFrequenciesPooling: SentenceRarity.WordFrequenciesPooling,
    val wordCounts: WordCounts = WordCounts(caseSensitive = false),
    val epsilon: Double = 1e-3f
) extends SentenceScore {
  override def name: String = {
    s"sr-$wordFrequenciesPooling-$wordCounts"
  }

  override def requiredSummaryScores: Seq[SummaryScore] = {
    Seq(wordCounts)
  }

  override def processSentence(
      language: Language,
      sentence: String,
      requiredValues: Seq[Float],
      requiredSummaries: Seq[SummaryScore]
  ): Float = {
    val logTotalCount = math.log(wordCounts.totalCount(language).toDouble)
    val counts = SentenceRarity.whitespaceRegex.split(sentence).map(word => {
      val count = wordCounts.count(language, word).toDouble
      if (count == 0.0)
        epsilon
      else
        count
    })
    val logFrequencies = counts.map(math.log(_) - logTotalCount)
    -wordFrequenciesPooling(logFrequencies, logFrequencies = true).toFloat
  }
}

object SentenceRarity {
  protected[SentenceRarity] val whitespaceRegex: Regex = "\\s+".r

  def apply(
      wordFrequenciesPooling: WordFrequenciesPooling,
      wordCounts: WordCounts = WordCounts(caseSensitive = false),
      epsilon: Double = 1e-3f
  ): SentenceRarity = {
    new SentenceRarity(wordFrequenciesPooling, wordCounts, epsilon)
  }

  trait WordFrequenciesPooling {
    def apply(frequencies: Seq[Double], logFrequencies: Boolean): Double
    override def toString: String
  }

  case object MinPooling extends WordFrequenciesPooling {
    override def apply(frequencies: Seq[Double], logFrequencies: Boolean): Double = {
      frequencies.min
    }

    override def toString: String = {
      "min"
    }
  }

  case object MaxPooling extends WordFrequenciesPooling {
    override def apply(frequencies: Seq[Double], logFrequencies: Boolean): Double = {
      frequencies.max
    }

    override def toString: String = {
      "max"
    }
  }

  case object MeanPooling extends WordFrequenciesPooling {
    override def apply(frequencies: Seq[Double], logFrequencies: Boolean): Double = {
      if (logFrequencies)
        logSumExp(frequencies)
      else
        frequencies.sum / frequencies.size.toDouble
    }

    override def toString: String = {
      "mean"
    }
  }

  case object ProductPooling extends WordFrequenciesPooling {
    override def apply(frequencies: Seq[Double], logFrequencies: Boolean): Double = {
      if (logFrequencies)
        frequencies.sum
      else
        frequencies.product
    }

    override def toString: String = {
      "prod"
    }
  }

  private def logSumExp(a: Double, b: Double): Double = {
    if (a == Double.NegativeInfinity) b
    else if (b == Double.NegativeInfinity) a
    else if (a < b) b + math.log(1 + math.exp(a - b))
    else a + math.log(1 + math.exp(b - a))
  }

  private def logSumExp(values: Seq[Double]): Double = {
    values.length match {
      case 0 => Double.NegativeInfinity
      case 1 => values(0)
      case 2 => logSumExp(values(0), values(1))
      case _ =>
        val max = values.max
        if (max.isInfinite) max
        else max + math.log(values.map(_ - max).sum)
    }
  }
}
