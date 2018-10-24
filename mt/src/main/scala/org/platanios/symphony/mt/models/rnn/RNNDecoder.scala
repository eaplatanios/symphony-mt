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

package org.platanios.symphony.mt.models.rnn

import org.platanios.symphony.mt.Environment
import org.platanios.symphony.mt.models.Model.DecodingMode
import org.platanios.symphony.mt.models._
import org.platanios.symphony.mt.models.parameters.ParameterManager
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple
import org.platanios.tensorflow.api.ops.seq2seq.decoders.{BasicDecoder, BeamSearchDecoder, GooglePenalty}

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class RNNDecoder[T: TF : IsNotQuantized, State: OutputStructure]()
    extends Decoder[T, (Tuple[Output[T], Seq[State]], Output[Int], Output[Int])] {
  override def create[O: TF](
      decodingMode: DecodingMode[O],
      config: RNNModel.Config[T, _],
      encoderState: (Tuple[Output[T], Seq[State]], Output[Int], Output[Int]),
      beginOfSequenceToken: String,
      endOfSequenceToken: String,
      tgtSequences: Output[Int] = null,
      tgtSequenceLengths: Output[Int] = null
  )(implicit
      stage: Stage,
      mode: Mode,
      env: Environment,
      parameterManager: ParameterManager,
      deviceManager: DeviceManager,
      context: Output[Int]
  ): RNNDecoder.DecoderOutput[O]

  protected def decode[O: TF, DecState: OutputStructure, DecStateShape](
      decodingMode: DecodingMode[O],
      config: RNNModel.Config[T, _],
      srcSequenceLengths: Output[Int],
      tgtSequences: Output[Int],
      tgtSequenceLengths: Output[Int],
      initialState: DecState,
      embeddings: Output[Int] => Output[T],
      cell: tf.RNNCell[Output[T], DecState, Shape, DecStateShape],
      tgtMaxLength: Output[Int],
      beginOfSequenceToken: String,
      endOfSequenceToken: String
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      context: Output[Int],
      evOutputToShapeDecState: OutputToShape.Aux[DecState, DecStateShape]
  ): RNNDecoder.DecoderOutput[O] = {
    val outputWeights = parameterManager.getProjectionToWords(cell.outputShape.apply(-1), context(1)).castTo[T]

    val output = decodingMode match {
      case Model.DecodingTrainMode =>
        // Time-major transpose
        val transposedSequences = if (config.timeMajor) tgtSequences.transpose() else tgtSequences
        val embeddedSequences = embeddings(transposedSequences)

        // Decoder RNN
        val helper = BasicDecoder.TrainingHelper[Output[T], DecState, Shape](
          input = embeddedSequences,
          sequenceLengths = tgtSequenceLengths,
          timeMajor = config.timeMajor)
        val decoder = BasicDecoder(cell, initialState, helper, outputLayer(outputWeights))
        val tuple = decoder.decode(
          outputTimeMajor = config.timeMajor,
          parallelIterations = config.env.parallelIterations,
          swapMemory = config.env.swapMemory)
        RNNDecoder.DecoderOutput(tuple._1.modelOutput, tuple._3)
      case Model.DecodingInferMode =>
        // Decoder embeddings
        val embeddingFn = (o: Output[Int]) => embeddings(o)
        val tgtVocabLookupTable = parameterManager.stringToIndexLookup(context(1))
        val tgtBosID = tgtVocabLookupTable(tf.constant(beginOfSequenceToken)).toInt
        val tgtEosID = tgtVocabLookupTable(tf.constant(endOfSequenceToken)).toInt

        val batchSize = tf.shape(srcSequenceLengths).slice(0).expandDims(0)

        // Decoder RNN
        if (config.beamWidth > 1) {
          val decoder = BeamSearchDecoder(
            cell, initialState, embeddingFn, tf.fill(INT32, batchSize)(tgtBosID),
            tgtEosID, config.beamWidth, GooglePenalty(config.lengthPenaltyWeight),
            outputLayer(outputWeights))
          val tuple = decoder.decode(
            outputTimeMajor = config.timeMajor, maximumIterations = tgtMaxLength,
            parallelIterations = config.env.parallelIterations, swapMemory = config.env.swapMemory)
          RNNDecoder.DecoderOutput(tuple._1.predictedIDs(---, 0), tuple._3(---, 0).toInt)
        } else {
          val decHelper = BasicDecoder.GreedyEmbeddingHelper[T, DecState](
            embeddingFn = embeddingFn,
            beginTokens = tf.fill(INT32, batchSize)(tgtBosID),
            endToken = tgtEosID)
          val decoder = BasicDecoder(cell, initialState, decHelper, outputLayer(outputWeights))
          val tuple = decoder.decode(
            outputTimeMajor = config.timeMajor, maximumIterations = tgtMaxLength,
            parallelIterations = config.env.parallelIterations, swapMemory = config.env.swapMemory)
          RNNDecoder.DecoderOutput(tuple._1.sample, tuple._3)
        }
    }
    output.asInstanceOf[RNNDecoder.DecoderOutput[O]]
  }

  protected def outputLayer(outputWeights: Output[T])(logits: Output[T]): Output[T] = {
    if (logits.rank == 3) {
      val reshapedLogits = tf.reshape(logits, Shape(-1, logits.shape(-1)))
      val product = tf.matmul(reshapedLogits, outputWeights)
      if (logits.shape(1) == -1 || outputWeights.shape(1) == -1) {
        tf.reshape(
          product,
          tf.concatenate(Seq(tf.shape(logits).slice(0 :: -1), tf.shape(outputWeights).slice(1, NewAxis)), axis = 0))
      } else {
        tf.reshape(product, logits.shape(0 :: -1) + outputWeights.shape(1))
      }
    } else {
      tf.matmul(logits, outputWeights)
    }
  }
}

object RNNDecoder {
  case class DecoderOutput[T](
      sequences: Output[T],
      lengths: Output[Int])
}
