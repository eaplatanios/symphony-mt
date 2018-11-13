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

package org.platanios.symphony.mt.data.statistics

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.data.{FileParallelDataset, newReader}

import better.files._

import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
trait SentenceStatistic[T] {
  protected val whitespaceRegex: Regex = "\\s+".r

  def forDatasets(datasets: Seq[FileParallelDataset]): Map[Language, T] = {
    datasets.flatMap(_.files.map(languageAndFiles => languageAndFiles._1 -> {
      aggregate(languageAndFiles._2.map(file => {
        forSentences(
          language = languageAndFiles._1,
          sentences = newReader(file).lines().toAutoClosedIterator.map(whitespaceRegex.split(_).toSeq).toSeq)
      }))
    })).groupBy(_._1).mapValues(values => aggregate(values.map(_._2)))
  }

  def forSentences(language: Language, sentences: Seq[Seq[String]]): T
  def aggregate(values: Seq[T]): T
}
