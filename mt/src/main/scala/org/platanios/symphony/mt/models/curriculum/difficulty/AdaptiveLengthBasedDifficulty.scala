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

package org.platanios.symphony.mt.models.curriculum.difficulty

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.data.scores.SentenceLengthsHistogram
import org.platanios.symphony.mt.data.{DataConfig, FileParallelDataset}
import org.platanios.symphony.mt.models.curriculum.difficulty.LengthBasedDifficulty._
import org.platanios.symphony.mt.models.{Context, SentencePairs}
import org.platanios.tensorflow.api._

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
class AdaptiveLengthBasedDifficulty[T](
    override val lengthSelector: LengthBasedDifficulty.LengthSelector,
    val datasets: Seq[FileParallelDataset],
    val languagePairs: Seq[(Language, Language)],
    val dataConfig: DataConfig
) extends LengthBasedDifficulty[T](lengthSelector) {
  protected val sentenceLengthsHistogram: SentenceLengthsHistogram = {
    // TODO: [CURRICULUM] What about when `srcMaxLength == -1` or `tgtMaxLength == -1`?
    val maxLength = lengthSelector match {
      case SourceLengthSelector => dataConfig.srcMaxLength
      case TargetLengthSelector => dataConfig.tgtMaxLength
    }
    new SentenceLengthsHistogram(minLength = 0, maxLength = maxLength)
  }

  protected val lengthHistograms: Map[Language, Seq[Int]] = {
    //    val filesPerLanguage = lengthSelector match {
    //      case SourceLengthSelector =>
    //        languagePairs.map(_._1).map(language => {
    //          language -> datasets.filter(_.languages.contains(language))
    //        }).toMap
    //      case TargetLengthSelector =>
    //        languagePairs.map(_._2).map(language => {
    //          language -> datasets.filter(_.languages.contains(language))
    //        }).toMap
    //    }
    val filesPerLanguage = languagePairs.flatMap(p => Seq(p._1, p._2)).map(language => {
      language -> datasets.filter(_.languages.contains(language))
    }).toMap
    filesPerLanguage.map {
      case (language, files) => language -> sentenceLengthsHistogram.forDatasets(language, files)
    }
  }

  protected val lengthCDFs: Map[Language, Seq[Float]] = {
    lengthHistograms.mapValues(values => {
      val cumulative = values.scanLeft(0)(_ + _).tail
      cumulative.map(_.toFloat / cumulative.last)
    })
  }

  protected var languages            : Seq[Language]                     = _
  protected val lengthCDFsLookupTable: mutable.Map[Graph, Output[Float]] = mutable.Map.empty

  protected def removeGraph(graph: Graph): Unit = {
    lengthCDFsLookupTable -= graph
  }

  override def initialize()(implicit context: Context): Unit = {
    languages = context.languages.map(_._1)
  }

  override def apply(sample: SentencePairs[T]): Output[Float] = {
    val (languageId, length) = lengthSelector match {
      case SourceLengthSelector => (sample._1._1, sample._2._2)
      case TargetLengthSelector => (sample._1._2, sample._3._2)
    }
    val table = lengthCDFsLookupTable.getOrElseUpdate(tf.currentGraph, {
      tf.nameScope("Difficulty/AdaptiveLengthBasedDifficulty") {
        tf.nameScope("LengthCDFs") {
          tf.constant[Float](languages.map(this.lengthCDFs(_).toTensor).toTensor)
        }
      }
    })
    table.gather(languageId).gather(length)
  }
}
