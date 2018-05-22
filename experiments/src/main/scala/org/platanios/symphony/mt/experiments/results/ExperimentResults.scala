///* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
// *
// * Licensed under the Apache License, Version 2.0 (the "License"); you may not
// * use this file except in compliance with the License. You may obtain a copy of
// * the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// * License for the specific language governing permissions and limitations under
// * the License.
// */
//
//package org.platanios.symphony.mt.experiments.results
//
//import org.platanios.symphony.mt.experiments.Metric
//
//import vegas._
//
///**
//  * @author Emmanouil Antonios Platanios
//  */
//object ExperimentResults {
//  def plot(
//      results: Seq[(String, String, Seq[ExperimentResult])],
//      metric: Metric,
//      datasets: Set[String],
//      datasetTags: Set[String],
//      evalDatasets: Set[String],
//      evalDatasetTags: Set[String],
//      title: String
//  ): Unit = {
//    val bestResults = results.map(r => Map(
//      "series" -> r._1,
//      "x" -> r._2,
//      "y" -> ExperimentResult.mean(
//        ExperimentResult.best(r._3)(metric, evalDatasets, evalDatasetTags)
//      )(metric, datasets, datasetTags)._2))
//    Vegas("A simple bar chart with embedded data.")
//        .withData(bestResults)
//        .encodeColumn("x", Ordinal, scale = Scale(padding = 2.0), axis = Axis(title = "#Parallel Sentences", orient = Orient.Bottom, axisWidth = 1.0, offset = -8.0))
//        .encodeX("series", Nominal, scale = Scale(bandSize = 50.0, padding = 1.0), sortOrder = SortOrder.Desc, hideAxis = true)
//        .encodeY("y", Quantitative, axis = Axis(title = "BLEU", grid = false))
//        .encodeColor("series", Nominal, scale = Scale(rangeNominals = List("#EA98D2", "#659CCA")))
//        .configFacet(cell = CellConfig(strokeWidth = 0.0))
//        .configLegend(symbolShape = "square")
//        .mark(Bar)
//        .window
//        .show
//  }
//}
