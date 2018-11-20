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
import org.platanios.symphony.mt.data.{newReader, newWriter}
import org.platanios.symphony.mt.utilities.Histogram

import better.files._

import scala.collection.mutable

// TODO: [DATA] [SCORES] Create a `LanguageSpecificSummaryScore` abstraction for this and the `WordCounts` summaries.

/**
  * @author Emmanouil Antonios Platanios
  */
class SentenceScoreHistogram(
    val score: SentenceScore,
    val maxNumBins: Int,
    val languageSpecific: Boolean = true
) extends SummaryScore {
  protected val histograms: mutable.HashMap[Language, Histogram] = {
    mutable.HashMap.empty[Language, Histogram]
  }

  override def name: String = {
    s"$score.$maxNumBins.bins.histogram"
  }

  override def requiredSentenceScores: Seq[SentenceScore] = {
    Seq(score)
  }

  override def processSentence(
      language: Language,
      sentence: String,
      requiredValues: Seq[Float],
      requiredSummaries: Seq[SummaryScore]
  ): Unit = {
    val sentenceScore = score.processSentence(language, sentence, requiredValues, requiredSummaries)
    histogram(language).insert(sentenceScore)
  }

  def cdfScore: SentenceScore = {
    val histogramScore = this
    new SentenceScore {
      override def name: String = {
        s"$histogramScore.cdf"
      }

      override def requiredSentenceScores: Seq[SentenceScore] = {
        Seq(score)
      }

      override def requiredSummaryScores: Seq[SummaryScore] = {
        Seq(histogramScore)
      }

      override def processSentence(
          language: Language,
          sentence: String,
          requiredValues: Seq[Float],
          requiredSummaries: Seq[SummaryScore]
      ): Float = {
        histogramScore.histogram(language).cdf(requiredValues.head).toFloat
      }
    }
  }

  override def resetState(): Unit = {
    histograms.clear()
  }

  override def saveStateToFile(file: File): Unit = {
    val writer = newWriter(file)
    if (languageSpecific) {
      histograms.foreach {
        case (language, histogram) =>
          writer.write(s"${SentenceScoreHistogram.FILE_LANGUAGE_SEPARATOR}\n")
          writer.write(s"${language.name}\n")
          histogram.bins.foreach(bin => {
            writer.write(s"${bin.mean}\t${bin.numSamples}\n")
          })
      }
    } else {
      histograms.values.head.bins.foreach(bin => {
        writer.write(s"${bin.mean}\t${bin.numSamples}\n")
      })
    }
    writer.flush()
    writer.close()
  }

  override def loadStateFromFile(file: File): Unit = {
    reset()
    if (languageSpecific) {
      var currentLanguage: Option[Language] = None
      newReader(file).lines().toAutoClosedIterator.foreach(line => {
        if (line == SentenceScoreHistogram.FILE_LANGUAGE_SEPARATOR) {
          currentLanguage = None
        } else {
          currentLanguage match {
            case None => currentLanguage = Some(Language.fromName(line))
            case Some(language) =>
              val lineParts = line.split('\t')
              histograms.getOrElseUpdate(language, Histogram(maxNumBins))
                  .insertBin(Histogram.Bin(mean = lineParts(0).toDouble, numSamples = lineParts(1).toLong))
          }
        }
      })
    } else {
      val histogram = Histogram(maxNumBins)
      newReader(file).lines().toAutoClosedIterator.foreach(line => {
        val lineParts = line.split('\t')
        histogram.insertBin(Histogram.Bin(mean = lineParts(0).toDouble, numSamples = lineParts(1).toLong))
      })
      histograms.update(null, histogram)
    }
  }

  /** Obtains the histogram to use for a specific language. */
  protected def histogram(language: Language): Histogram = {
    if (languageSpecific) {
      histograms.getOrElseUpdate(language, Histogram(maxNumBins))
    } else if (histograms.isEmpty) {
      histograms.getOrElseUpdate(language, Histogram(maxNumBins))
    } else {
      // This is a little hack in case we are using the same histogram for all languages.
      histograms.values.head
    }
  }
}

object SentenceScoreHistogram {
  private[SentenceScoreHistogram] val FILE_LANGUAGE_SEPARATOR: String = "-" * 100

  def apply(
      score: SentenceScore,
      maxNumBins: Int
  ): SentenceScoreHistogram = {
    new SentenceScoreHistogram(score, maxNumBins)
  }
}
