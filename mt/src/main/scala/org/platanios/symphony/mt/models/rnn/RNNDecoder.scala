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

import org.platanios.symphony.mt.models.{ModelConstructionContext, Sequences}
import org.platanios.symphony.mt.models.Transformation.Decoder
import org.platanios.symphony.mt.models.decoders.{BasicDecoder, BeamSearchDecoder, OutputLayer}
import org.platanios.symphony.mt.models.helpers.Common
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape}
import org.platanios.tensorflow.api.tf.RNNCell

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class RNNDecoder[T: TF : IsNotQuantized, State, DecState: OutputStructure, DecStateShape](
    val numUnits: Int,
    val outputLayer: OutputLayer
)(implicit
    protected val evOutputToShapeDecState: OutputToShape.Aux[DecState, DecStateShape]
) extends Decoder[EncodedSequences[T, State]] {
  override def applyTrain(
      encodedSequences: EncodedSequences[T, State]
  )(implicit context: ModelConstructionContext): Sequences[Float] = {
    // TODO: What if no target sequences are provided?
    val tgtSequences = context.tgtSequences.get

    // Shift the target sequence one step forward so the decoder learns to output the next word.
    val tgtBosId = tf.constant[Int](Vocabulary.BEGIN_OF_SEQUENCE_TOKEN_ID)
    val batchSize = tf.shape(tgtSequences.sequences).slice(0)
    val shiftedTgtSequences = tf.concatenate(Seq(
      tf.fill[Int, Int](tf.stack[Int](Seq(batchSize, 1)))(tgtBosId),
      tgtSequences.sequences), axis = 1)
    val shiftedTgtSequenceLengths = tgtSequences.lengths + 1
    val (cell, initialState) = cellAndInitialState(encodedSequences, Some(tgtSequences))

    // Embed the target sequences.
    val embeddedTgtSequences = embeddings(shiftedTgtSequences)

    // Decoder RNN
    val helper = BasicDecoder.TrainingHelper[Output[T], DecState, Shape](
      input = embeddedTgtSequences,
      sequenceLengths = shiftedTgtSequenceLengths,
      timeMajor = false)
    val decoder = BasicDecoder(cell, initialState, helper, outputLayer[T](numUnits))
    val tuple = decoder.decode(
      outputTimeMajor = false,
      parallelIterations = context.env.parallelIterations,
      swapMemory = context.env.swapMemory)
    Sequences(tuple._1.modelOutput.toFloat, tuple._3)
  }

  override def applyInfer(
      encodedSequences: EncodedSequences[T, State]
  )(implicit context: ModelConstructionContext): Sequences[Int] = {
    val (cell, initialState) = cellAndInitialState(encodedSequences)

    // Determine the maximum allowed sequence length to consider while decoding.
    val maxDecodingLength = {
      if (!context.mode.isTraining && context.dataConfig.tgtMaxLength != -1)
        tf.constant(context.dataConfig.tgtMaxLength)
      else
        tf.round(tf.max(encodedSequences.lengths).toFloat *
            context.inferenceConfig.maxDecodingLengthFactor).toInt
    }

    // Create some constants that will be used during decoding.
    val tgtBosID = tf.constant[Int](Vocabulary.BEGIN_OF_SEQUENCE_TOKEN_ID)
    val tgtEosID = tf.constant[Int](Vocabulary.END_OF_SEQUENCE_TOKEN_ID)

    // Create the decoder RNN.
    val batchSize = tf.shape(encodedSequences.lengths).slice(0).expandDims(0)
    val embeddings = (ids: Output[Int]) => this.embeddings(ids)
    val output = {
      if (context.inferenceConfig.beamWidth > 1) {
        val decoder = BeamSearchDecoder(
          cell, initialState, embeddings, tf.fill[Int, Int](batchSize)(tgtBosID),
          tgtEosID, context.inferenceConfig.beamWidth, context.inferenceConfig.lengthPenalty,
          outputLayer[T](cell.outputShape.apply(-1)))
        val tuple = decoder.decode(
          outputTimeMajor = false,
          maximumIterations = maxDecodingLength,
          parallelIterations = context.env.parallelIterations,
          swapMemory = context.env.swapMemory)
        Sequences(tuple._1.predictedIDs(---, 0), tuple._3(---, 0).toInt)
      } else {
        val decHelper = BasicDecoder.GreedyEmbeddingHelper[T, DecState](
          embeddingFn = embeddings,
          beginTokens = tf.fill[Int, Int](batchSize)(tgtBosID),
          endToken = tgtEosID)
        val decoder = BasicDecoder(cell, initialState, decHelper, outputLayer[T](cell.outputShape.apply(-1)))
        val tuple = decoder.decode(
          outputTimeMajor = false,
          maximumIterations = maxDecodingLength,
          parallelIterations = context.env.parallelIterations,
          swapMemory = context.env.swapMemory)
        Sequences(tuple._1.sample, tuple._3)
      }
    }

    // Make sure the outputs are of shape [batchSize, time] or [beamWidth, batchSize, time]
    // when using beam search.
    val outputSequences = {
      if (output.sequences.rank == 3)
        output.sequences.transpose(Tensor(2, 0, 1))
      else
        output.sequences
    }
    Sequences(outputSequences(---, 0 :: -1), output.lengths - 1)
  }

  protected def embeddings(
      ids: Output[Int]
  )(implicit context: ModelConstructionContext): Output[T] = {
    context.parameterManager.wordEmbeddings(context.tgtLanguageID, ids).castTo[T]
  }

  protected def cellAndInitialState(
      encodedSequences: EncodedSequences[T, State],
      tgtSequences: Option[Sequences[Int]] = None
  )(implicit context: ModelConstructionContext): (RNNCell[Output[T], DecState, Shape, DecStateShape], DecState)
}
