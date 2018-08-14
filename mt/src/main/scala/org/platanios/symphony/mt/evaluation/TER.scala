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

package org.platanios.symphony.mt.evaluation

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.utilities.Encoding
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.ops.metrics.Metric._

import ter.core.{CostFunction, TerScorer}

/**
  * @author Emmanouil Antonios Platanios
  */
class TER protected (
    val normalize: Boolean = false,
    val ignoreCase: Boolean = true,
    val ignorePunctuation: Boolean = false,
    val asianLanguagesSupport: Boolean = false,
    val ignoreHTMLTagBrackets: Boolean = false,
    val beamWidth: Int = 20,
    val maximumShiftDistance: Int = 50,
    val shiftCost: Float = 1.0f,
    val matchCost: Float = 0.0f,
    val deleteCost: Float = 1.0f,
    val insertCost: Float = 1.0f,
    val substituteCost: Float = 1.0f,
    val variablesCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
    val valuesCollections: Set[Graph.Key[Output]] = Set(METRIC_VALUES),
    val updatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
    val resetsCollections: Set[Graph.Key[Op]] = Set(METRIC_RESETS),
    override val name: String = "TER"
)(implicit languages: Seq[(Language, Vocabulary)]) extends MTMetric {
  protected val terScorer: TerScorer = new TerScorer()
  terScorer.setNormalize(normalize)
  terScorer.setCase(!ignoreCase)
  terScorer.setPunct(ignorePunctuation)
  terScorer.setAsian(asianLanguagesSupport)
  terScorer.setTagBrackets(ignoreHTMLTagBrackets)
  terScorer.setBeamWidth(beamWidth)
  terScorer.setShiftDist(maximumShiftDistance)

  protected val terCostFunction: CostFunction = new CostFunction()
  terCostFunction._shift_cost = shiftCost
  terCostFunction._match_cost = matchCost
  terCostFunction._delete_cost = deleteCost
  terCostFunction._insert_cost = insertCost
  terCostFunction._substitute_cost = substituteCost

  // TODO: Do not ignore the weights.

  private[this] def counts(batch: Seq[Tensor[INT32]]): Seq[Tensor[FLOAT32]] = {
    val (tgtLanguageId, hyp, hypLen, ref, refLen) = (batch(0), batch(1), batch(2), batch(3), batch(4))
    val tgtLanguage = tgtLanguageId.scalar

    val (hypSentences, hypLengths) = (hyp.unstack(), hypLen.unstack())
    val (refSentences, refLengths) = (ref.unstack(), refLen.unstack())
    val hypSeq = hypSentences.zip(hypLengths).map {
      case (s, len) =>
        val lenScalar = len.scalar
        val seq = s(0 :: lenScalar).entriesIterator.map(v => Encoding.tfStringToUTF8(v.asInstanceOf[String])).toSeq
        languages(tgtLanguage)._2.decodeSequence(seq)
    }
    val refSeq = refSentences.zip(refLengths).map {
      case (s, len) =>
        val lenScalar = len.scalar
        val seq = s(0 :: lenScalar).entriesIterator.map(v => Encoding.tfStringToUTF8(v.asInstanceOf[String])).toSeq
        languages(tgtLanguage)._2.decodeSequence(seq)
    }

    var totalEdits = 0.0f
    var totalWords = 0.0f
    hypSeq.zip(refSeq).foreach {
      case (hypothesis, reference) =>
        val result = terScorer.TER(hypothesis.mkString(" "), reference.mkString(" "), terCostFunction)
        totalEdits += result.numEdits.toFloat
        totalWords += result.numWords.toFloat
    }

    Seq(Tensor(totalEdits), Tensor(totalWords))
  }

  override def compute(
      values: ((Output, Output, Output), (Output, Output)),
      weights: Option[Output] = None,
      name: String = this.name
  ): Output = {
    val ((tgtLanguageId, src, srcLen), (tgt, tgtLen)) = values
    var ops = Set(src.op, srcLen.op, tgt.op, tgtLen.op)
    weights.foreach(ops += _.op)
    tf.createWithNameScope(name, ops) {
      val _counts = tf.callback(
        counts, Seq(tgtLanguageId, src, srcLen, tgt, tgtLen), Seq(FLOAT32, FLOAT32), stateful = false)
      100 * tf.divide(_counts(0), _counts(1), name = "Value")
    }
  }

  override def streaming(
      values: ((Output, Output, Output), (Output, Output)),
      weights: Option[Output] = None,
      name: String = this.name
  ): Metric.StreamingInstance[Output] = {
    val ((tgtLanguageId, src, srcLen), (tgt, tgtLen)) = values
    var ops = Set(src.op, srcLen.op, tgt.op, tgtLen.op)
    weights.foreach(ops += _.op)
    tf.variableScope(name) {
      tf.createWithNameScope(name, ops) {
        val totalEdits = variable("TotalEdits", FLOAT32, Shape(), tf.ZerosInitializer, variablesCollections)
        val totalWords = variable("TotalWords", FLOAT32, Shape(), tf.ZerosInitializer, variablesCollections)
        val _counts = tf.callback(
          counts, Seq(tgtLanguageId, src, srcLen, tgt, tgtLen), Seq(FLOAT32, FLOAT32), stateful = false)
        val updateTotalEdits = totalEdits.assignAdd(_counts(0))
        val updateTotalWords = totalWords.assignAdd(_counts(1))
        val value = 100 * tf.divide(totalEdits.value, totalWords.value, name = "Value")
        val update = 100 * tf.divide(updateTotalEdits, updateTotalWords, name = "Update")
        val reset = tf.group(Set(totalEdits.initializer, totalWords.initializer), name = "Reset")
        valuesCollections.foreach(tf.currentGraph.addToCollection(value, _))
        updatesCollections.foreach(tf.currentGraph.addToCollection(update, _))
        resetsCollections.foreach(tf.currentGraph.addToCollection(reset, _))
        Metric.StreamingInstance(value, update, reset, Set(totalEdits, totalWords))
      }
    }
  }
}

object TER {
  def apply(
      normalize: Boolean = false,
      ignoreCase: Boolean = true,
      ignorePunctuation: Boolean = false,
      asianLanguagesSupport: Boolean = false,
      ignoreHTMLTagBrackets: Boolean = false,
      beamWidth: Int = 20,
      maximumShiftDistance: Int = 50,
      shiftCost: Float = 1.0f,
      matchCost: Float = 0.0f,
      deleteCost: Float = 1.0f,
      insertCost: Float = 1.0f,
      substituteCost: Float = 1.0f,
      variablesCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
      valuesCollections: Set[Graph.Key[Output]] = Set(METRIC_VALUES),
      updatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
      resetsCollections: Set[Graph.Key[Op]] = Set(METRIC_RESETS),
      name: String = "TER"
  )(implicit languages: Seq[(Language, Vocabulary)]): TER = {
    new TER(
      normalize, ignoreCase, ignorePunctuation, asianLanguagesSupport, ignoreHTMLTagBrackets, beamWidth,
      maximumShiftDistance, shiftCost, matchCost, deleteCost, insertCost, substituteCost, variablesCollections,
      valuesCollections, updatesCollections, resetsCollections, name)(languages)
  }
}
