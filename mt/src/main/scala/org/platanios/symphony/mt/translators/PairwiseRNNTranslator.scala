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
import org.platanios.symphony.mt.translators.PairwiseTranslator.{MTInferLayer, MTLossLayer, MTTrainLayer}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.layers.LayerInstance
import org.platanios.tensorflow.api.learn.layers.rnn.cell.RNNCell
import org.platanios.tensorflow.api.ops.rnn.decoder.BasicRNNDecoder

/**
  * @author Emmanouil Antonios Platanios
  */
class PairwiseRNNTranslator[S, SS](
    val dataType: DataType = FLOAT32,
    val sourceEmbeddingSize: Int = 300,
    val targetEmbeddingSize: Int = 300,
    val encoderCell: tf.learn.RNNCell[Output, Shape, S, SS],
    val decoderCell: tf.learn.RNNCell[Output, Shape, S, SS],
    override protected var configuration: Configuration = Configuration()
)(implicit
    evSupportedS: ops.rnn.cell.RNNCell.Supported.Aux[S, SS]
) extends PairwiseTranslator(configuration) {
  private[this] def encoder(
      input: (Output, Output),
      sourceVocabularySize: Int,
      variableFn: (String, DataType, Shape) => Variable,
      mode: Mode
  ): (RNNCell.Tuple[Output, S], Set[Variable], Set[Variable]) = {
    val encEmbeddings = variableFn("EncoderEmbeddings", dataType, Shape(sourceVocabularySize, sourceEmbeddingSize))
    val encEmbeddedInput = tf.embeddingLookup(encEmbeddings, input._1)
    val encCellInstance = encoderCell.createCell(encEmbeddedInput, mode)
    val encTuple = tf.dynamicRNN(
      encCellInstance.cell, encEmbeddedInput, timeMajor = true,
      parallelIterations = configuration.parallelIterations, swapMemory = configuration.swapMemory,
      sequenceLengths = input._2)
    (encTuple, encCellInstance.trainableVariables + encEmbeddings, encCellInstance.nonTrainableVariables)
  }

  private[this] def decoder(
      input: (Output, Output),
      tgtVocabSize: Int,
      tgtVocab: tf.LookupTable,
      encTuple: RNNCell.Tuple[Output, S],
      variableFn: (String, DataType, Shape) => Variable,
      mode: Mode
  ): (((Output, Output), S, Output), Set[Variable], Set[Variable]) = {
    val decEmbeddings = variableFn("DecoderEmbeddings", dataType, Shape(tgtVocabSize, targetEmbeddingSize))
    val decEmbeddingFn = tf.embeddingLookup(decEmbeddings, _)
    val decCellInstance = decoderCell.createCell(encTuple.output, mode)
    val decHelper = BasicRNNDecoder.GreedyEmbeddingHelper[S](
      decEmbeddingFn, tf.fill(dataType, Shape(configuration.batchSize))(tgtVocab.lookup(configuration.beginOfSequenceToken)),
      tgtVocab.lookup(configuration.endOfSequenceToken))
    val decTuple = BasicRNNDecoder(decCellInstance.cell, encTuple.state, decHelper).dynamicDecode(
      outputTimeMajor = true, maximumIterations = tf.round(tf.max(input._2) * 2),
      parallelIterations = configuration.parallelIterations,
      swapMemory = configuration.swapMemory)
    (decTuple, decCellInstance.trainableVariables + decEmbeddings, decCellInstance.nonTrainableVariables)
  }

  private[this] def trainDecoder(
      input: (Output, Output, Output),
      tgtVocabSize: Int,
      encTuple: RNNCell.Tuple[Output, S],
      variableFn: (String, DataType, Shape) => Variable,
      mode: Mode
  ): (((Output, Output), S, Output), Set[Variable], Set[Variable]) = {
    val decEmbeddings = variableFn("DecoderEmbeddings", dataType, Shape(tgtVocabSize, targetEmbeddingSize))
    val decEmbeddedInput = tf.embeddingLookup(decEmbeddings, input._1)
    val decCellInstance = decoderCell.createCell(encTuple.output, mode)
    val decHelper = BasicRNNDecoder.TrainingHelper(decEmbeddedInput, input._3, timeMajor = true)
    val decTuple = BasicRNNDecoder(decCellInstance.cell, encTuple.state, decHelper).dynamicDecode(
      outputTimeMajor = true, parallelIterations = configuration.parallelIterations,
      swapMemory = configuration.swapMemory)
    (decTuple, decCellInstance.trainableVariables + decEmbeddings, decCellInstance.nonTrainableVariables)
  }

  private[this] def output(
      logits: Output,
      lengths: Output,
      tgtVocabSize: Int,
      variableFn: (String, DataType, Shape) => Variable
  ): ((Output, Output), Set[Variable], Set[Variable]) = {
    val outWeights = variableFn("OutputWeights", dataType, Shape(logits.shape(-1), tgtVocabSize))
    val outBias = variableFn("OutputBias", dataType, Shape(tgtVocabSize))
    val output = tf.matmul(logits, outWeights) + outBias
    ((output, lengths), Set(outWeights, outBias), Set.empty)
  }

  override protected def trainLayer(
      srcVocabSize: Int,
      tgtVocabSize: Int,
      srcVocab: tf.LookupTable,
      tgtVocab: tf.LookupTable
  ): MTTrainLayer = {
    new tf.learn.Layer[((Output, Output), (Output, Output, Output)), (Output, Output)]("PairwiseRNNTranslation") {
      override val layerType: String = "PairwiseRNNTranslation"

      override def forward(
          input: ((Output, Output), (Output, Output, Output)),
          mode: Mode
      ): LayerInstance[((Output, Output), (Output, Output, Output)), (Output, Output)] = {
        val variableFn = variable(_, _, _)
        val (encTuple, encTrVars, encNonTrVars) = encoder(input._1, srcVocabSize, variableFn, mode)
        val (decTuple, decTrVars, decNonTrVars) = trainDecoder(input._2, tgtVocabSize, encTuple, variableFn, mode)
        val (out, outTrVars, outNonTrVars) = output(decTuple._1._1, decTuple._3, tgtVocabSize, variableFn)
        LayerInstance(input, out, encTrVars ++ decTrVars ++ outTrVars, encNonTrVars ++ decNonTrVars ++ outNonTrVars)
      }
    }
  }

  override protected def inferLayer(
      srcVocabSize: Int,
      tgtVocabSize: Int,
      srcVocab: tf.LookupTable,
      tgtVocab: tf.LookupTable
  ): MTInferLayer = {
    new tf.learn.Layer[(Output, Output), (Output, Output)]("PairwiseRNNTranslation") {
      override val layerType: String = "PairwiseRNNTranslation"

      override def forward(input: (Output, Output), mode: Mode): LayerInstance[(Output, Output), (Output, Output)] = {
        val variableFn = variable(_, _, _)
        val (encTuple, encTrVars, encNonTrVars) = encoder(input, srcVocabSize, variableFn, mode)
        val (decTuple, decTrVars, decNonTrVars) = decoder(input, tgtVocabSize, tgtVocab, encTuple, variableFn, mode)
        val (out, outTrVars, outNonTrVars) = output(decTuple._1._1, decTuple._3, tgtVocabSize, variableFn)
        LayerInstance(input, out, encTrVars ++ decTrVars ++ outTrVars, encNonTrVars ++ decNonTrVars ++ outNonTrVars)
      }
    }
  }

  override protected def lossLayer(): MTLossLayer = {
    new tf.learn.Layer[((Output, Output), (Output, Output, Output)), Output]("PairwiseRNNTranslationLoss") {
      override val layerType: String = "PairwiseRNNTranslationLoss"

      override def forward(
          input: ((Output, Output), (Output, Output, Output)),
          mode: Mode
      ): LayerInstance[((Output, Output), (Output, Output, Output)), Output] = {
        val loss = tf.sequenceLoss(
          input._1._1, input._2._2,
          weights = tf.sequenceMask(input._1._2, dataType = input._1._1.dataType),
          averageAcrossTimeSteps = false, averageAcrossBatch = true)
        LayerInstance(input, loss)
      }
    }
  }

  override protected def optimizer(): tf.train.Optimizer = tf.train.GradientDescent(1.0)
}
