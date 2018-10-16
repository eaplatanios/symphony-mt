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
import org.platanios.symphony.mt.models.{Sentences, SentencesWithLanguage, SentencesWithLanguagePair}
import org.platanios.symphony.mt.utilities.Encoding
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
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
    val variablesCollections: Set[Graph.Key[Variable[Any]]] = Set(METRIC_VARIABLES),
    val valuesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_VALUES),
    val updatesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_UPDATES),
    val resetsCollections: Set[Graph.Key[UntypedOp]] = Set(METRIC_RESETS),
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

  private def counts(
      tgtLanguageId: Tensor[Int],
      hypotheses: Tensor[String],
      hypothesisLengths: Tensor[Int],
      references: Tensor[String],
      referenceLengths: Tensor[Int]
  ): (Tensor[Float], Tensor[Float]) = {
    val tgtLanguage = tgtLanguageId.scalar
    val (hypSentences, hypLengths) = (hypotheses.unstack(), hypothesisLengths.unstack())
    val (refSentences, refLengths) = (references.unstack(), referenceLengths.unstack())
    val hypSeq = hypSentences.zip(hypLengths).map {
      case (s, len) =>
        val lenScalar = len.scalar
        val seq = s(0 :: lenScalar).entriesIterator.map(v => Encoding.tfStringToUTF8(v)).toSeq
        languages(tgtLanguage)._2.decodeSequence(seq)
    }
    val refSeq = refSentences.zip(refLengths).map {
      case (s, len) =>
        val lenScalar = len.scalar
        val seq = s(0 :: lenScalar).entriesIterator.map(v => Encoding.tfStringToUTF8(v)).toSeq
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

    (Tensor(totalEdits), Tensor(totalWords))
  }

  override def compute(
      values: (SentencesWithLanguage[String], (SentencesWithLanguagePair[String], Sentences[String])),
      weights: Option[Output[Float]] = None,
      name: String = this.name
  ): Output[Float] = {
    val tgtLanguageId = values._1._1
    val hypSentences = values._1._2._1
    val hypSentenceLengths = values._1._2._2
    val refSentences = values._2._2._1
    val refSentenceLengths = values._2._2._2
    tf.nameScope(name) {
      val _counts = tf.callback(
        function = Function.tupled(counts _),
        input = (tgtLanguageId, hypSentences, hypSentenceLengths, refSentences, refSentenceLengths),
        outputDataType = (FLOAT32, FLOAT32),
        stateful = false,
        name = "Counts")
      Output.constant[Float](100.0f) * tf.divide(_counts._1, _counts._2, name = "Value")
    }
  }

  override def streaming(
      values: (SentencesWithLanguage[String], (SentencesWithLanguagePair[String], Sentences[String])),
      weights: Option[Output[Float]] = None,
      name: String = this.name
  ): Metric.StreamingInstance[Output[Float]] = {
    val tgtLanguageId = values._1._1
    val hypSentences = values._1._2._1
    val hypSentenceLengths = values._1._2._2
    val refSentences = values._2._2._1
    val refSentenceLengths = values._2._2._2
    tf.variableScope(name) {
      tf.nameScope(name) {
        val totalEdits = variable[Float]("TotalEdits", Shape(), tf.ZerosInitializer, variablesCollections)
        val totalWords = variable[Float]("TotalWords", Shape(), tf.ZerosInitializer, variablesCollections)
        val _counts = tf.callback(
          function = Function.tupled(counts _),
          input = (tgtLanguageId, hypSentences, hypSentenceLengths, refSentences, refSentenceLengths),
          outputDataType = (FLOAT32, FLOAT32),
          stateful = false,
          name = "Counts")
        val updateTotalEdits = totalEdits.assignAdd(_counts._1)
        val updateTotalWords = totalWords.assignAdd(_counts._2)
        val value = 100.0f * tf.divide(totalEdits.value, totalWords.value, name = "Value")
        val update = 100.0f * tf.divide(updateTotalEdits, updateTotalWords, name = "Update")
        val reset = tf.group(Set(totalEdits.initializer, totalWords.initializer), name = "Reset")
        valuesCollections.foreach(tf.currentGraph.addToCollection(_)(value.asUntyped))
        updatesCollections.foreach(tf.currentGraph.addToCollection(_)(update.asUntyped))
        resetsCollections.foreach(tf.currentGraph.addToCollection(_)(reset.asUntyped))
        Metric.StreamingInstance(value, update, reset, Set(totalEdits.asUntyped, totalWords.asUntyped))
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
      variablesCollections: Set[Graph.Key[Variable[Any]]] = Set(METRIC_VARIABLES),
      valuesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_VALUES),
      updatesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_UPDATES),
      resetsCollections: Set[Graph.Key[UntypedOp]] = Set(METRIC_RESETS),
      name: String = "TER"
  )(implicit languages: Seq[(Language, Vocabulary)]): TER = {
    new TER(
      normalize, ignoreCase, ignorePunctuation, asianLanguagesSupport, ignoreHTMLTagBrackets, beamWidth,
      maximumShiftDistance, shiftCost, matchCost, deleteCost, insertCost, substituteCost, variablesCollections,
      valuesCollections, updatesCollections, resetsCollections, name)(languages)
  }
}
