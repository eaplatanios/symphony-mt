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

package org.platanios.symphony.mt.config

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.data.FileParallelDataset
import org.platanios.symphony.mt.evaluation.{MTMetric, SentenceCount, SentenceLength}

/**
  * @author Emmanouil Antonios Platanios
  */
case class EvaluationConfig(
    frequency: Int = 1000,
    metrics: Seq[MTMetric] = Seq(
      SentenceLength(forHypothesis = true, name = "HypLen"),
      SentenceLength(forHypothesis = false, name = "RefLen"),
      SentenceCount(name = "#Sentences")),
    datasets: Seq[(String, FileParallelDataset)] = Seq.empty,
    languagePairs: Set[(Language, Language)] = Set.empty)
