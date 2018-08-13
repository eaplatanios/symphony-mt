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

package org.platanios.symphony.mt.experiments

import better.files._

import java.nio.charset.StandardCharsets

import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
object LogParser {
  protected[LogParser] val evaluationStepRegex : Regex = """.*Learn / Hooks / Evaluation - Step (.*) Evaluation:""".r
  protected[LogParser] val evaluationStartRegex: Regex = """.*Learn / Hooks / Evaluation - ╔[═╤]*╗""".r
  protected[LogParser] val evaluationStopRegex : Regex = """.*Learn / Hooks / Evaluation - ╚[═╧]*╝""".r
  protected[LogParser] val evaluationLineRegex : Regex = """.*Learn / Hooks / Evaluation - ║ ([^│]*) │ (.*) ║""".r

  def parseEvaluationResults(file: File): Seq[ExperimentResult] = {
    var results = Seq.empty[ExperimentResult]
    var step = -1L
    var metrics = Seq.empty[String]
    file.lineIterator(StandardCharsets.UTF_8).foreach(line => {
      if (step == -1L) {
        evaluationStepRegex.findFirstMatchIn(line) match {
          case Some(m) => step = m.group(1).toLong
          case None => ()
        }
      } else {
        if (evaluationStopRegex.findFirstMatchIn(line).nonEmpty) {
          step = -1L
          metrics = Seq.empty[String]
        } else {
          evaluationLineRegex.findFirstMatchIn(line) match {
            case Some(m) if metrics.isEmpty =>
              metrics = m.group(2).split('│').map(_.trim)
            case Some(m) =>
              val dataset = m.group(1).split('/')
              results ++= m.group(2).split('│').map(_.trim.toDouble).zip(metrics).map(p => {
                ExperimentResult(Metric.fromHeader(p._2), step, dataset(0), dataset(1), dataset(2), p._1)
              })
            case None => ()
          }
        }
      }
    })
    results
  }
}
