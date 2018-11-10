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

package org.platanios.symphony.mt.models.parameters

import org.platanios.symphony.mt.models.{Context, parameters}
import org.platanios.tensorflow.api._

/**
  * @author Emmanouil Antonios Platanios
  */
class PairwiseManager protected (
    override val wordEmbeddingsType: WordEmbeddingsType,
    override val variableInitializer: tf.VariableInitializer = null
) extends ParameterManager(wordEmbeddingsType, variableInitializer) {
  override def get[P: TF](
      name: String,
      shape: Shape,
      variableInitializer: tf.VariableInitializer = variableInitializer,
      variableReuse: tf.VariableReuse = tf.ReuseOrCreateNewVariable
  )(implicit context: Context): Output[P] = {
    tf.variableScope("ParameterManager") {
      val graph = currentGraph

      // Determine all the language index pairs that are relevant given the current model configuration.
      val languageIndexPairs = parameters.languageIndexPairs(context.languages.map(_._1), context.modelConfig)
      val languagePairs = languageIndexPairs.map(p => (context.languages(p._1)._1, context.languages(p._2)._1))
      val languageIdPairs = languageIndexPairs.map(p => (languageIds(graph)(p._1), languageIds(graph)(p._2)))

      // Obtain the variable values for all language pairs.
      val variableValues = languagePairs.map(pair => {
        tf.variable[P](
          s"$name/${pair._1.abbreviation}-${pair._2.abbreviation}", shape,
          initializer = variableInitializer, reuse = variableReuse).value
      })

      // Choose the variable for the current language pair.
      tf.nameScope(name) {
        val predicates = variableValues.zip(languageIdPairs).map {
          case (v, (srcLangId, tgtLangId)) =>
            (tf.logicalAnd(
              tf.equal(context.srcLanguageID, srcLangId),
              tf.equal(context.tgtLanguageID, tgtLangId)), () => v)
        }
        val assertion = tf.assert(
          tf.any(tf.stack(predicates.map(_._1))),
          Seq(
            Output[String]("No variables found for the provided language pair."),
            Output[String]("Context source language: "), context.srcLanguageID,
            Output[String]("Context target language: "), context.tgtLanguageID))
        val default = () => tf.createWith(controlDependencies = Set(assertion)) {
          tf.identity(variableValues.head)
        }
        tf.cases(predicates, default)
      }
    }
  }
}

object PairwiseManager {
  def apply(
      wordEmbeddingsType: WordEmbeddingsType,
      variableInitializer: tf.VariableInitializer = null
  ): PairwiseManager = {
    new PairwiseManager(wordEmbeddingsType, variableInitializer)
  }
}
