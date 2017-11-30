/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.symphony.mt.translators

import org.platanios.symphony.mt.core.Configuration
import org.platanios.symphony.mt.translators.PairwiseTranslator.{MTInferLayer, MTTrainLayer}
import org.platanios.tensorflow.api.learn.{EVALUATION, INFERENCE, Mode}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.layers.LayerInstance
import org.platanios.tensorflow.api.ops.rnn.decoder.{BasicRNNDecoder, BeamSearchRNNDecoder}
import org.platanios.tensorflow.api.ops.variables.Initializer

/**
  * @author Emmanouil Antonios Platanios
  */
class PairwiseRNNTranslator[S, SS](
    val dataType: DataType = FLOAT32,
    val encoderCell: tf.learn.RNNCell[Output, Shape, S, SS],
    val decoderCell: tf.learn.RNNCell[Output, Shape, S, SS],
    override val configuration: Configuration = Configuration()
)(implicit
    evSWhileLoopVariable: ops.control_flow.WhileLoopVariable.Aux[S, SS],
    evSBeamSearchRNNDecoderSupported: BeamSearchRNNDecoder.Supported.Aux[S, SS]
) extends PairwiseTranslator(configuration) {
  private[this] def encoder(
      input: (Output, Output),
      sourceVocabularySize: Int,
      variableFn: (String, DataType, Shape, Initializer) => Variable,
      mode: Mode
  ): (tf.learn.RNNTuple[Output, S], Set[Variable], Set[Variable]) = tf.createWithNameScope("Encoder") {
    val encEmbeddings = variableFn(
      "EncoderEmbeddings", dataType, Shape(sourceVocabularySize, configuration.modelNumUnits),
      tf.RandomUniformInitializer(-0.1f, 0.1f))
    val encEmbeddedInput = tf.embeddingLookup(encEmbeddings, input._1)
    val encCellInstance = encoderCell.createCell(mode)
    val encTuple = tf.dynamicRNN[Output, Shape, S, SS](
      encCellInstance.cell, encEmbeddedInput, timeMajor = false,
      parallelIterations = configuration.parallelIterations, swapMemory = configuration.swapMemory,
      sequenceLengths = input._2)
    (encTuple, encCellInstance.trainableVariables + encEmbeddings, encCellInstance.nonTrainableVariables)
  }

  private[this] def decoder(
      input: (Output, Output),
      tgtVocabSize: Int,
      tgtVocab: tf.LookupTable,
      encTuple: tf.learn.RNNTuple[Output, S],
      variableFn: (String, DataType, Shape, Initializer) => Variable,
      mode: Mode
  ): ((Output, Output), Set[Variable], Set[Variable]) = {
    tf.createWithNameScope("Decoder") {
      val bosToken = tf.constant(configuration.beginOfSequenceToken)
      val eosToken = tf.constant(configuration.endOfSequenceToken)
      val decEmbeddings = variableFn(
        "DecoderEmbeddings", dataType, Shape(tgtVocabSize, configuration.modelNumUnits),
        tf.RandomUniformInitializer(-0.1f, 0.1f))
      val decEmbeddingFn = (o: Output) => tf.embeddingLookup(decEmbeddings, o)
      val decCellInstance = decoderCell.createCell(mode)
      val decOutputLayer = decoderOutputLayer(decCellInstance.cell.outputShape(-1), tgtVocabSize, variableFn)
      if (configuration.inferBeamWidth > 1) {
        val decoder = BeamSearchRNNDecoder(
          decCellInstance.cell, encTuple.state, decEmbeddingFn,
          tf.fill(INT32, tf.shape(input._2))(tgtVocab.lookup(bosToken)), tgtVocab.lookup(eosToken).cast(INT32),
          configuration.inferBeamWidth, configuration.inferLengthPenaltyWeight, decOutputLayer._1)
        val decTuple = decoder.decode(
          outputTimeMajor = false, maximumIterations = inferMaxLength(tf.max(input._2)),
          parallelIterations = configuration.parallelIterations,
          swapMemory = configuration.swapMemory)
        ((decTuple._1.predictedIDs, decTuple._3),
            decCellInstance.trainableVariables ++ decOutputLayer._2 + decEmbeddings,
            decCellInstance.nonTrainableVariables)
      } else {
        val decHelper = BasicRNNDecoder.GreedyEmbeddingHelper[S](
          decEmbeddingFn,
          tf.fill(INT32, tf.shape(input._2))(tgtVocab.lookup(bosToken)),
          tgtVocab.lookup(eosToken).cast(INT32))
        val decoder = BasicRNNDecoder(decCellInstance.cell, encTuple.state, decHelper, decOutputLayer._1)
        val decTuple = decoder.decode(
          outputTimeMajor = false, maximumIterations = inferMaxLength(tf.max(input._2)),
          parallelIterations = configuration.parallelIterations,
          swapMemory = configuration.swapMemory)
        ((decTuple._1.sample, decTuple._3),
            decCellInstance.trainableVariables ++ decOutputLayer._2 + decEmbeddings,
            decCellInstance.nonTrainableVariables)
      }
    }
  }

  private[this] def trainDecoder(
      input: (Output, Output, Output),
      tgtVocabSize: Int,
      encTuple: tf.learn.RNNTuple[Output, S],
      variableFn: (String, DataType, Shape, Initializer) => Variable,
      mode: Mode
  ): ((BasicRNNDecoder.Output[Output, Shape], S, Output), Set[Variable], Set[Variable]) = {
    tf.createWithNameScope("TrainDecoder") {
      val decEmbeddings = variableFn(
        "DecoderEmbeddings", dataType, Shape(tgtVocabSize, configuration.modelNumUnits),
        tf.RandomUniformInitializer(-0.1f, 0.1f))
      val decEmbeddedInput = tf.embeddingLookup(decEmbeddings, input._1)
      val decCellInstance = decoderCell.createCell(mode)
      val decHelper = BasicRNNDecoder.TrainingHelper(decEmbeddedInput, input._3, timeMajor = false)
      val decOutputLayer = decoderOutputLayer(decCellInstance.cell.outputShape(-1), tgtVocabSize, variableFn)
      val decoder = BasicRNNDecoder(decCellInstance.cell, encTuple.state, decHelper, decOutputLayer._1)
      val decTuple = decoder.decode(
        outputTimeMajor = false, parallelIterations = configuration.parallelIterations,
        swapMemory = configuration.swapMemory)
      (decTuple,
          decCellInstance.trainableVariables ++ decOutputLayer._2 + decEmbeddings,
          decCellInstance.nonTrainableVariables)
    }
  }

  private[this] def decoderOutputLayer(
      decOutputDepth: Int,
      tgtVocabSize: Int,
      variableFn: (String, DataType, Shape, Initializer) => Variable
  ): (Output => Output, Set[Variable]) = {
    val w = variableFn(
      "OutWeights", dataType, Shape(decOutputDepth, tgtVocabSize), tf.RandomUniformInitializer(-0.1f, 0.1f))
    val layer = (logits: Output) => {
      if (logits.rank > 2) {
        // Broadcasting is required for the inputs.
        val product = tf.tensorDot(logits, w.value, Seq(logits.rank - 1), Seq(0))
        // Reshape the output back to the original rank of the input.
        product.setShape(logits.shape(0 :: -1) + tgtVocabSize)
        product
      } else {
        tf.matmul(logits, w.value)
      }
    }
    (layer, Set(w))
  }

  private[this] def output(
      decTuple:(BasicRNNDecoder.Output[Output, Shape], S, Output),
      tgtVocabSize: Int,
      mode: Mode
  ): (Output, Output) = {
    val lengths = decTuple._3
    mode match {
      case INFERENCE | EVALUATION => (decTuple._1.sample, lengths)
      case _ => (decTuple._1.rnnOutput, lengths)
    }
  }

  override protected def trainLayer(
      srcVocabSize: Int,
      tgtVocabSize: Int,
      srcVocab: () => tf.LookupTable,
      tgtVocab: () => tf.LookupTable
  ): MTTrainLayer = {
    new tf.learn.Layer[((Output, Output), (Output, Output, Output)), (Output, Output)]("PairwiseRNNTranslation") {
      override val layerType: String = "PairwiseRNNTranslation"

      override def forward(
          input: ((Output, Output), (Output, Output, Output)),
          mode: Mode
      ): LayerInstance[((Output, Output), (Output, Output, Output)), (Output, Output)] = {
        val variableFn: (String, DataType, Shape, Initializer) => Variable = variable(_, _, _, _)
        val (encTuple, encTrVars, encNonTrVars) = encoder(input._1, srcVocabSize, variableFn, mode)
        val (decTuple, decTrVars, decNonTrVars) = trainDecoder(input._2, tgtVocabSize, encTuple, variableFn, mode)
        val out = output(decTuple, tgtVocabSize, mode)
        LayerInstance(input, out, encTrVars ++ decTrVars, encNonTrVars ++ decNonTrVars)
      }
    }
  }

  override protected def inferLayer(
      srcVocabSize: Int,
      tgtVocabSize: Int,
      srcVocab: () => tf.LookupTable,
      tgtVocab: () => tf.LookupTable
  ): MTInferLayer = {
    new tf.learn.Layer[(Output, Output), (Output, Output)]("PairwiseRNNTranslation") {
      override val layerType: String = "PairwiseRNNTranslation"

      override def forward(input: (Output, Output), mode: Mode): LayerInstance[(Output, Output), (Output, Output)] = {
        val variableFn: (String, DataType, Shape, Initializer) => Variable = variable(_, _, _, _)
        srcVocab()
        val (encTuple, encTrVars, encNonTrVars) = encoder(input, srcVocabSize, variableFn, mode)
        val (out, decTrVars, decNonTrVars) = decoder(input, tgtVocabSize, tgtVocab(), encTuple, variableFn, mode)
        LayerInstance(input, out, encTrVars ++ decTrVars, encNonTrVars ++ decNonTrVars)
      }
    }
  }
}
