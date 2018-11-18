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

  private var hash: Option[String] = None

  override val isSummary: Boolean = true

  private[scores] def setDatasetsHash(hash: String): Unit = {
    this.hash = Some(hash)
  }

  private[scores] def setDatasetsHashFromFile(file: File): Unit = {
    setDatasetsHash(file.extension(includeDot = false).get)
  }

  private[scores] def reset(): Unit = {
    hash = None
    resetState()
  }

  protected def resetState(): Unit

  def saveStateToFile(file: File): Unit
  def loadStateFromFile(file: File): Unit

  override def toString: String = {
    hash.map(h => s"$name.$h").getOrElse(name)
  }
}

object Score {
  private val logger = Logger(LoggerFactory.getLogger("Data / Scorer"))

  @throws[IllegalStateException]
  def scoreDatasets(
      datasets: Seq[FileParallelDataset],
      score: Score,
      sentenceScoreNamesFileSuffix: String = ".sentence.scores.names",
      sentenceScoreValuesFileSuffix: String = ".sentence.scores.values",
      summaryScoresDir: Option[File] = None,
      alwaysRecompute: Boolean = false
  ): Unit = {
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
          summaryScore.setDatasetsHash(datasets.hashCode.toHexString)
          Seq(summaryScore)
        } else {
          scores.takeWhile(!_.isSummary)
        }
      }

      val (summaryScoreFile, computeSummaryScore) = {
        if (computingSummaryScore) {
          val summaryScore = scoresToCompute.head.asInstanceOf[SummaryScore]
          val summaryScoreFile = summaryScoresDir.map(_ / s"$summaryScore")
          if (alwaysRecompute || summaryScoreFile.isEmpty || summaryScoreFile.get.notExists) {
            (summaryScoreFile, Some(true))
          } else {
            (summaryScoreFile, Some(false))
          }
        } else {
          (None, None)
        }
      }

      logger.info(s"Computing scores: ${scoresToCompute.map(s => s"'$s'").mkString(", ")}.")
      datasets.foreach(dataset => {
        dataset.files.foreach {
          case (language, files) => files.foreach(file => {
            val fileNameWithoutExtension = file.nameWithoutExtension(includeAll = false)
            val scoreNamesFile = file.sibling(fileNameWithoutExtension + s"$sentenceScoreNamesFileSuffix")
            val scoreValuesFile = file.sibling(fileNameWithoutExtension + s"$sentenceScoreValuesFileSuffix")

            var sentenceScoreNames = {
              if (scoreNamesFile.notExists)
                Seq.empty
              else
                readScoreNames(scoreNamesFile)
            }

            var sentenceScoreValues = {
              if (scoreValuesFile.notExists)
                Seq.empty
              else
                readScoreValues(scoreValuesFile).map(v => mutable.ArrayBuffer(v: _*))
            }

            if (computingSummaryScore) {
              val summaryScore = scoresToCompute.head.asInstanceOf[SummaryScore]

              if (computeSummaryScore.get) {
                val requiredSentenceScores = summaryScore.requiredSentenceScores.map(s => {
                  val index = sentenceScoreNames.indexOf(s.toString)
                  sentenceScoreValues(index)
                })

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
                  scoresToCompute.filter(s => !sentenceScoreNames.contains(s.toString))
              }

              if (sentenceScoresToCompute.nonEmpty) {
                val scoreNames = sentenceScoresToCompute.map(_.toString)
                scoreNames.foreach(name => {
                  val index = sentenceScoreNames.indexOf(name)
                  if (index >= 0) {
                    sentenceScoreNames = sentenceScoreNames.take(index) ++ sentenceScoreNames.drop(index + 1)
                    sentenceScoreValues = sentenceScoreValues.take(index) ++ sentenceScoreValues.drop(index + 1)
                  }
                })

                val indexOffset = sentenceScoreValues.size

                sentenceScoreNames ++= scoreNames
                sentenceScoreValues ++= sentenceScoresToCompute.map(_ => mutable.ArrayBuffer.empty[Float])

                val scoresWithRequirements = sentenceScoresToCompute.map(score => {
                  val requiredSentenceScores = score.requiredSentenceScores.map(s => {
                    val index = sentenceScoreNames.indexOf(s.toString)
                    sentenceScoreValues(index)
                  })

                  val requiredSummaryScores = score.requiredSummaryScores.map(s => {
                    computedSummaryScores.find(_ == s).get
                  })

                  (score, requiredSentenceScores, requiredSummaryScores)
                })

                // Compute the new scores for all sentences.
                newReader(file).lines().toAutoClosedIterator.zipWithIndex.foreach(sentence => {
                  scoresWithRequirements.zipWithIndex.map(score => {
                    val sentenceScore = score._1._1.processSentence(
                      language = language,
                      sentence = sentence._1,
                      requiredValues = score._1._2.map(_.apply(sentence._2)),
                      requiredSummaries = score._1._3)
                    sentenceScoreValues(indexOffset + score._2) += sentenceScore.asInstanceOf[Float]
                  })
                })

                writeScoreNames(scoreNamesFile, sentenceScoreNames)
                writeScoreValues(scoreValuesFile, sentenceScoreValues)

                logger.info(s"Updated score names file '$scoreNamesFile' and score values file '$scoreValuesFile'.")
              }
            }
          })
        }
      })

      if (computingSummaryScore) {
        val summaryScore = scoresToCompute.head.asInstanceOf[SummaryScore]

        summaryScoreFile.foreach(file => {
          if (computeSummaryScore.get) {
            // Save state to file, if necessary.
            if (!file.exists)
              file.parent.createDirectories()
            summaryScore.saveStateToFile(file)
          } else {
            // Load state from file.
            summaryScore.loadStateFromFile(file)
            summaryScore.setDatasetsHashFromFile(file)
          }
        })

        computedSummaryScores += summaryScore
      }

      scores = scores.drop(scoresToCompute.size)
    }
  }

  private def writeScoreNames(
      file: File,
      names: Seq[String],
      binaryFormat: Boolean = false
  ): Unit = {
    val writer = newWriter(file)
    writer.write(names.mkString("\t"))
    writer.flush()
    writer.close()
  }

  private def readScoreNames(
      file: File,
      binaryFormat: Boolean = false
  ): Seq[String] = {
    newReader(file).lines().toAutoClosedIterator.next().split('\t')
  }

  private def writeScoreValues(
      file: File,
      values: Seq[Seq[Float]],
      binaryFormat: Boolean = false
  ): Unit = {
    val writer = newWriter(file)
    values.transpose.foreach(v => writer.write(s"${v.mkString("\t")}\n"))
    writer.flush()
    writer.close()
  }

  private def readScoreValues(
      file: File,
      binaryFormat: Boolean = false
  ): Seq[Seq[Float]] = {
    newReader(file).lines().toAutoClosedIterator.map(line => {
      line.split('\t').map(_.toFloat)
    }).toSeq.transpose
  }
}
