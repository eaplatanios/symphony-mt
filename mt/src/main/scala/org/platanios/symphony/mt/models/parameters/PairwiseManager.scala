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

/**
  * @author Emmanouil Antonios Platanios
  */
class PairwiseManager protected (
    override val wordEmbeddingsType: WordEmbeddingsType,
    override val variableInitializer: tf.VariableInitializer = null
) extends ParameterManager(wordEmbeddingsType, variableInitializer) {
  override def get(
      name: String,
      dataType: DataType,
      shape: Shape,
      variableInitializer: tf.VariableInitializer = variableInitializer,
      variableReuse: tf.VariableReuse = tf.ReuseOrCreateNewVariable
  )(implicit stage: Stage): Output = {
    tf.variableScope("ParameterManager") {
      val graph = currentGraph

      // Obtain the variable values for all language pairs.
      val variableValues = languages.map(_._1)
          .combinations(2)
          .map(c => (c(0), c(1)))
          .flatMap(p => Seq(p, (p._2, p._1)))
          .toSeq
          .map(pair => {
            tf.variable(
              s"$name/${pair._1.abbreviation}-${pair._2.abbreviation}", dataType, shape,
              initializer = variableInitializer, reuse = variableReuse).value
          })

      // Choose the variable for the current language pair.
      tf.createWithNameScope(name) {
        val languageIdPairs = languageIds(graph)
            .combinations(2)
            .map(c => (c(0), c(1)))
            .flatMap(p => Seq(p, (p._2, p._1)))
            .toSeq
        val predicates = variableValues.zip(languageIdPairs).map {
          case (v, (srcLangId, tgtLangId)) =>
            (tf.logicalAnd(
              tf.equal(context.get._1, srcLangId),
              tf.equal(context.get._2, tgtLangId)), () => v)
        }
        val assertion = tf.assert(false, Seq("No variables found for the provided language pair."))
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
