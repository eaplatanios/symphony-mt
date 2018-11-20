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
import org.platanios.symphony.mt.data.{FileParallelDataset, newReader, newWriter}
import org.platanios.symphony.mt.utilities.TopologicalSort

import better.files._
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
trait Score {
  type T

  val isSummary: Boolean

  def name: String

  def requiredSentenceScores: Seq[SentenceScore] = {
    Seq.empty
  }

  def requiredSummaryScores: Seq[SummaryScore] = {
    Seq.empty
  }

  def processSentence(
      language: Language,
      sentence: String,
      requiredValues: Seq[Float],
      requiredSummaries: Seq[SummaryScore]
  ): T

  override def toString: String = {
    name
  }
}

trait SentenceScore extends Score {
  override type T = Float

  override val isSummary: Boolean = false
}

trait SummaryScore extends Score {
  override type T = Unit

  override val isSummary: Boolean = true

  private[scores] def reset(): Unit = {
    resetState()
  }

  protected def resetState(): Unit

  def saveStateToFile(file: File): Unit
  def loadStateFromFile(file: File): Unit
}

object Score {
  private val logger = Logger(LoggerFactory.getLogger("Data / Scorer"))

  @throws[IllegalStateException]
  def scoreDatasets(
      datasets: Seq[FileParallelDataset],
      score: Score,
      scoresDir: File,
      alwaysRecompute: Boolean = false
  ): Unit = {
    val summaryScoresDir = scoresDir / "summary_scores"
    val sentenceScoresDir = scoresDir / "sentence_scores"

    summaryScoresDir.createDirectories()
    sentenceScoresDir.createDirectories()

    var computedSummaryScores = Set.empty[SummaryScore]
    var scores = TopologicalSort.sort[Score](
      values = Set(score),
      requirements = (s: Score) => s.requiredSentenceScores.toSet ++ s.requiredSummaryScores.toSet
    ).getOrElse(throw new IllegalStateException("There should be no cycles in the scores dependencies."))
    while (scores.nonEmpty) {
      val computingSummaryScore = scores.head.isSummary
      val scoresToCompute = {
        if (computingSummaryScore) {
          val summaryScore = scores.head.asInstanceOf[SummaryScore]
          summaryScore.reset()
          logger.info(s"Computing summary score '$summaryScore'.")
          Seq(summaryScore)
        } else {
          val sentenceScores = scores.takeWhile(!_.isSummary)
          logger.info(s"Computing sentence scores: ${sentenceScores.map(s => s"'$s'").mkString(", ")}.")
          sentenceScores
        }
      }

      val (summaryScoreFile, computeSummaryScore) = {
        if (computingSummaryScore) {
          val summaryScore = scoresToCompute.head.asInstanceOf[SummaryScore]
          val summaryScoreFile = summaryScoresDir / s"$summaryScore.score"
          if (alwaysRecompute || summaryScoreFile.notExists) {
            (summaryScoreFile, true)
          } else {
            (summaryScoreFile, false)
          }
        } else {
          (null, false)
        }
      }

      datasets.foreach(dataset => {
        dataset.files.foreach {
          case (language, files) => files.foreach(file => {
            val baseFilename = file.path.toAbsolutePath.normalize.toString.replace('/', '_')
            val scoreFiles = sentenceScoresDir.collectChildren(_.name.startsWith(baseFilename)).toArray

            val scoreNames = scoreFiles.map(_.name.drop(baseFilename.length + 1).dropRight(6))
            val scoreValues = scoreFiles.map(file => {
              mutable.ArrayBuffer(newReader(file).lines().toAutoClosedIterator.map(_.toFloat).toSeq: _*)
            })

            var sentenceScores = mutable.HashMap(scoreNames.zip(scoreValues): _*)

            if (computingSummaryScore) {
              val summaryScore = scoresToCompute.head.asInstanceOf[SummaryScore]

              if (computeSummaryScore) {
                val requiredSentenceScores = summaryScore.requiredSentenceScores.map(s => sentenceScores(s.toString))
                val requiredSummaryScores = summaryScore.requiredSummaryScores.map(s => {
                  computedSummaryScores.find(_ == s).get
                })

                // Update the summary score using all sentences.
                newReader(file).lines().toAutoClosedIterator.zipWithIndex.foreach(sentence => {
                  summaryScore.processSentence(
                    language = language,
                    sentence = sentence._1,
                    requiredValues = requiredSentenceScores.map(_.apply(sentence._2)),
                    requiredSummaries = requiredSummaryScores)
                })
              }
            } else {
              // Determine which scores need to be computed/re-computed.
              val sentenceScoresToCompute = {
                if (alwaysRecompute)
                  scoresToCompute
                else
                  scoresToCompute.filter(s => !sentenceScores.contains(s.toString))
              }

              if (sentenceScoresToCompute.nonEmpty) {
                // Remove the previously computed values for the scores that need to be computed now.
                sentenceScores ++= sentenceScoresToCompute.map(_.toString)
                    .zip(sentenceScoresToCompute.map(_ => mutable.ArrayBuffer.empty[Float]))

                val scoresWithRequirements = sentenceScoresToCompute.map(score => {
                  val requiredSentenceScores = score.requiredSentenceScores.map(s => sentenceScores(s.toString))
                  val requiredSummaryScores = score.requiredSummaryScores.map(s => {
                    computedSummaryScores.find(_ == s).get
                  })

                  (score, requiredSentenceScores, requiredSummaryScores)
                })

                // Compute the new scores for all sentences.
                newReader(file).lines().toAutoClosedIterator.zipWithIndex.foreach(sentence => {
                  scoresWithRequirements.zipWithIndex.foreach(score => {
                    val sentenceScore = score._1._1.processSentence(
                      language = language,
                      sentence = sentence._1,
                      requiredValues = score._1._2.map(_.apply(sentence._2)),
                      requiredSummaries = score._1._3)
                    sentenceScores(score._1._1.toString) :+= sentenceScore.asInstanceOf[Float]
                  })
                })

                sentenceScoresToCompute.foreach(score => {
                  val file = sentenceScoresDir / s"$baseFilename.$score.score"
                  val writer = newWriter(file)
                  sentenceScores(score.toString).foreach(v => writer.write(s"$v\n"))
                  writer.flush()
                  writer.close()
                  logger.info(s"Wrote sentence scores file '$file'.")
                })
              }
            }
          })
        }
      })

      if (computingSummaryScore) {
        val summaryScore = scoresToCompute.head.asInstanceOf[SummaryScore]

        if (computeSummaryScore) {
          // Save state to file, if necessary.
          if (!summaryScoreFile.exists)
            summaryScoreFile.parent.createDirectories()
          summaryScore.saveStateToFile(summaryScoreFile)
          logger.info(s"Wrote summary scores file '$summaryScoreFile'.")
        } else {
          // Load state from file.
          summaryScore.loadStateFromFile(summaryScoreFile)
        }

        computedSummaryScores += summaryScore
      }

      scores = scores.drop(scoresToCompute.size)
    }
  }
}
