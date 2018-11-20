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

package org.platanios.symphony.mt

import org.platanios.tensorflow.api._

/**
  * @author Emmanouil Antonios Platanios
  */
package object models {
  case class Sequences[T](
      sequences: Output[T],
      lengths: Output[Int])

  // Core Types

  type LanguageID = Output[Int]
  type LanguagePair = (LanguageID, LanguageID)
  type Sentences[T] = (Output[T], Output[Int])
  type SentencesWithLanguage[T] = (LanguageID, Sentences[T])
  type SentencesWithLanguagePair[T] = (LanguageID, LanguageID, Sentences[T])
  type SentencePairs[T] = (LanguagePair, Sentences[T], Sentences[T])

  type SentencesWithLanguageValue = (Tensor[Int], Tensor[String], Tensor[Int])
  type SentencesWithLanguagePairValue = (Tensor[Int], Tensor[Int], Tensor[String], Tensor[Int])

  // Estimators

  type TranslationEstimator = tf.learn.Estimator[
      /* In       */ SentencesWithLanguagePair[String],
      /* TrainIn  */ (SentencesWithLanguagePair[String], Sentences[String]),
      /* Out      */ SentencesWithLanguage[String],
      /* TrainOut */ SentencesWithLanguage[Float],
      /* Loss     */ Float,
      /* EvalIn   */ (SentencesWithLanguage[String], (SentencesWithLanguagePair[String], Sentences[String]))]
}
