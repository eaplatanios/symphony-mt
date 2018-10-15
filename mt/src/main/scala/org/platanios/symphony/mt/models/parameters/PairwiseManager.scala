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

import org.platanios.symphony.mt.models.Stage
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.TF

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
  )(implicit stage: Stage, context: Output[Int]): Output[P] = {
    tf.variableScope("ParameterManager") {
      val graph = currentGraph

      // Obtain the variable values for all language pairs.
      val variableValues = languages.map(_._1)
          .combinations(2)
          .map(c => (c(0), c(1)))
          .flatMap(p => Seq(p, (p._2, p._1)))
          .toSeq
          .map(pair => {
            tf.variable[P](
              s"$name/${pair._1.abbreviation}-${pair._2.abbreviation}", shape,
              initializer = variableInitializer, reuse = variableReuse).value
          })

      // Choose the variable for the current language pair.
      tf.nameScope(name) {
        val languageIdPairs = languageIds(graph)
            .combinations(2)
            .map(c => (c(0), c(1)))
            .flatMap(p => Seq(p, (p._2, p._1)))
            .toSeq
        val predicates = variableValues.zip(languageIdPairs).map {
          case (v, (srcLangId, tgtLangId)) =>
            (tf.logicalAnd(
              tf.equal(context(0), srcLangId),
              tf.equal(context(1), tgtLangId)), () => v)
        }
        val assertion = tf.assert(
          tf.any(tf.stack(predicates.map(_._1))),
          Seq(
            Output[String]("No variables found for the provided language pair."),
            Output[String]("Context source language: "), context(0),
            Output[String]("Context target language: "), context(1)))
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
