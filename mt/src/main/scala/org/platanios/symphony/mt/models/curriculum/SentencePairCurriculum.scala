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

package org.platanios.symphony.mt.models.curriculum

import org.platanios.symphony.mt.data.scores.SentenceScore
import org.platanios.symphony.mt.models.SentencePairsWithScores
import org.platanios.symphony.mt.models.curriculum.competency.Competency
import org.platanios.tensorflow.api.Output

/**
  * @author Emmanouil Antonios Platanios
  */
class SentencePairCurriculum(
    override val competency: Competency[Output[Float]],
    override val score: SentenceScore,
    val scoreSelector: SentencePairCurriculum.ScoreSelector,
    override val maxNumHistogramBins: Int = 1000
) extends DifficultyBasedCurriculum[SentencePairsWithScores[String]](
  competency = competency,
  score = score,
  maxNumHistogramBins = maxNumHistogramBins
) {
  override protected def sampleScore(sample: SentencePairsWithScores[String]): Output[Float] = {
    scoreSelector match {
      case SentencePairCurriculum.SourceSentenceScore => sample._2._3.get
      case SentencePairCurriculum.TargetSentenceScore => sample._3._3.get
    }
  }
}

object SentencePairCurriculum {
  sealed trait ScoreSelector
  object SourceSentenceScore extends ScoreSelector
  object TargetSentenceScore extends ScoreSelector

  def apply(
      competency: Competency[Output[Float]],
      score: SentenceScore,
      scoreSelector: ScoreSelector,
      maxNumHistogramBins: Int = 1000
  ): SentencePairCurriculum = {
    new SentencePairCurriculum(competency, score, scoreSelector, maxNumHistogramBins)
  }
}
