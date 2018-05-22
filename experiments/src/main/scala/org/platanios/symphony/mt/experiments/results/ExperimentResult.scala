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

package org.platanios.symphony.mt.experiments.results

import org.platanios.symphony.mt.experiments.Metric

/**
  * @author Emmanouil Antonios Platanios
  */
case class ExperimentResult(
    metric: Metric,
    step: Long,
    dataset: String,
    datasetTag: String,
    languagePair: String,
    value: Double)

object ExperimentResult {
  def best(results: Seq[ExperimentResult])(
      metric: Metric,
      datasets: Set[String] = results.map(_.dataset).toSet,
      datasetTags: Set[String] = results.map(_.datasetTag).toSet,
      languagePairs: Set[String] = results.map(_.languagePair).toSet
  ): Seq[ExperimentResult] = {
    val bestStep = results.groupBy(_.step).maxBy {
      case (_, stepResults) =>
        val filteredResults = stepResults.filter(r =>
          r.metric == metric &&
              datasets.contains(r.dataset) &&
              datasetTags.contains(r.datasetTag) &&
              languagePairs.contains(r.languagePair))
        val filteredValues = filteredResults.map(_.value).filter(!_.isNaN)
        filteredValues.sum
    }
    bestStep._2
  }

  def mean(results: Seq[ExperimentResult])(
      metric: Metric,
      datasets: Set[String] = results.map(_.dataset).toSet,
      datasetTags: Set[String] = results.map(_.datasetTag).toSet,
      languagePairs: Set[String] = results.map(_.languagePair).toSet
  ): (Long, Double) = {
    val filteredResults = results.filter(r =>
      r.metric == metric &&
          datasets.contains(r.dataset) &&
          datasetTags.contains(r.datasetTag) &&
          languagePairs.contains(r.languagePair))
    val filteredValues = filteredResults.map(_.value).filter(!_.isNaN)
    (results.head.step, filteredValues.sum / filteredValues.size)
  }
}
