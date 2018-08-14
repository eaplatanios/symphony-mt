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

package org.platanios.symphony.mt.models.pivoting

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.models.parameters.ParameterManager
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._

/**
  * @author Emmanouil Antonios Platanios
  */
case class SinglePivot(
    pivotLanguage: Language,
    supportedPairs: Seq[(Language, Language)]
) extends Pivot {
  protected var pivotLanguageId         : Output                = _
  protected var supportedLanguagePairIds: Seq[(Output, Output)] = _

  override def initialize(
      languages: Seq[(Language, Vocabulary)],
      parameterManager: ParameterManager
  ): Unit = {
    pivotLanguageId = parameterManager.languageId(languages.map(_._1).indexOf(pivotLanguage))
    supportedLanguagePairIds = supportedPairs.map {
      case (src, tgt) =>
        (parameterManager.languageId(languages.map(_._1).indexOf(src)),
            parameterManager.languageId(languages.map(_._1).indexOf(tgt)))
    }
  }

  override def pivotingSequence(srcLanguage: Output, tgtLanguage: Output): Output = {
    tf.cases(
      predicateFnPairs = supportedLanguagePairIds.map(lp =>
        (tf.logicalAnd(tf.equal(lp._1, srcLanguage), tf.equal(lp._2, tgtLanguage)), () => tgtLanguage(NewAxis))),
      default = () => tf.stack(Seq(pivotLanguageId, tgtLanguage)))
  }
}
