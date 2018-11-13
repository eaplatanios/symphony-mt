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
import org.platanios.symphony.mt.data.statistics.StatisticsCollector.Implicits._

import better.files._

import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
class StatisticsCollector[Statistics, Value](
    val statistics: Statistics
)(implicit evStatistics: StatisticsCollector.Statistics.Aux[Statistics, Value]) {
  protected val whitespaceRegex: Regex = "\\s+".r

  def collectForSentences(datasets: Seq[FileParallelDataset]): Map[Language, Value] = {
    datasets.flatMap(_.files.map(languageAndFiles => languageAndFiles._1 -> {
      statistics.aggregate(languageAndFiles._2.map(file => {
        statistics.aggregate(newReader(file).lines().toAutoClosedIterator.toSeq
            .map(line => {
              val sentence = whitespaceRegex.split(line)
              statistics.forSentence(languageAndFiles._1, sentence)
            }))
      }))
    })).groupBy(_._1).mapValues(values => statistics.aggregate(values.map(_._2)))
  }
}

object StatisticsCollector {
  object Implicits {
    implicit class RichStatistics[T, V](statistics: T)(implicit evStatistics: Statistics.Aux[T, V]) {
      def forSentence(language: Language, sentence: Seq[String]): V = {
        evStatistics.forSentence(statistics, language, sentence)
      }

      def aggregate(values: Seq[V]): V = {
        evStatistics.aggregate(statistics, values)
      }
    }
  }

  trait Statistics[T] {
    type Value

    def forSentence(statistic: T, language: Language, sentence: Seq[String]): Value
    def aggregate(statistic: T, values: Seq[Value]): Value
  }

  object Statistics {
    type Aux[T, V] = Statistics[T] {
      type Value = V
    }

    implicit def evSentenceStatistic[T]: Statistics.Aux[SentenceStatistic[T], T] = {
      new Statistics[SentenceStatistic[T]] {
        override type Value = T

        override def forSentence(statistic: SentenceStatistic[T], language: Language, sentence: Seq[String]): T = {
          statistic.forSentence(language, sentence)
        }

        override def aggregate(statistic: SentenceStatistic[T], values: Seq[T]): T = {
          statistic.aggregate(values)
        }
      }
    }
  }
}
