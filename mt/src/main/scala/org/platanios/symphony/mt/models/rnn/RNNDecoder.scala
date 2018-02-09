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
import org.platanios.symphony.mt.models.Decoder
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.cell.{RNNCell, Tuple}
import org.platanios.tensorflow.api.ops.seq2seq.decoders.{BasicDecoder, BeamSearchDecoder, GooglePenalty}

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class RNNDecoder[S, SS](
    val timeMajor: Boolean = false,
    // Inference
    val beamWidth: Int = 10,
    val lengthPenaltyWeight: Float = 0.0f,
    val decoderMaxLengthFactor: Float = 2.0f
)(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) extends Decoder[Tuple[Output, Seq[S]]] {
  def create(
      env: Environment,
      encoderTuple: Tuple[Output, Seq[S]],
      srcSequenceLengths: Output,
      tgtVocab: Vocabulary,
      tgtMaxLength: Int,
      beginOfSequenceToken: String,
      endOfSequenceToken: String,
      tgtSequences: Output = null,
      tgtSequenceLengths: Output = null,
      mode: Mode
  ): RNNDecoder.Output

  protected def decode[DS, DSS](
      env: Environment,
      srcSequenceLengths: Output,
      tgtSequences: Output,
      tgtSequenceLengths: Output,
      initialState: DS,
      embeddings: Variable,
      cell: RNNCell[Output, Shape, DS, DSS],
      tgtVocab: Vocabulary,
      tgtMaxLength: Int,
      beginOfSequenceToken: String,
      endOfSequenceToken: String,
      mode: Mode
  )(implicit
      evS: WhileLoopVariable.Aux[DS, DSS]
  ): RNNDecoder.Output = {
    val outputWeights = tf.variable(
      "OutWeights", embeddings.dataType, Shape(cell.outputShape(-1), tgtVocab.size),
      tf.RandomUniformInitializer(-0.1f, 0.1f))
    val outputLayer = (logits: Output) => tf.linear(logits, outputWeights.value)
    if (mode.isTraining) {
      // Time-major transpose
      val transposedSequences = if (timeMajor) tgtSequences.transpose() else tgtSequences
      val embeddedSequences = tf.embeddingLookup(embeddings, transposedSequences)

      // Decoder RNN
      val helper = BasicDecoder.TrainingHelper(embeddedSequences, tgtSequenceLengths, timeMajor)
      val decoder = BasicDecoder(cell, initialState, helper, outputLayer)
      val tuple = decoder.decode(
        outputTimeMajor = timeMajor, parallelIterations = env.parallelIterations,
        swapMemory = env.swapMemory)
      RNNDecoder.Output(tuple._1.rnnOutput, tuple._3)
    } else {
      // Decoder embeddings
      val embeddingFn = (o: Output) => tf.embeddingLookup(embeddings, o)
      val tgtVocabLookupTable = tgtVocab.lookupTable()
      val tgtBosID = tgtVocabLookupTable.lookup(tf.constant(beginOfSequenceToken)).cast(INT32)
      val tgtEosID = tgtVocabLookupTable.lookup(tf.constant(endOfSequenceToken)).cast(INT32)
      val maxDecodingLength = {
        if (tgtMaxLength != -1)
          tf.constant(tgtMaxLength)
        else
          tf.round(tf.max(tf.max(srcSequenceLengths)) * decoderMaxLengthFactor).cast(INT32)
      }

      // Decoder RNN
      if (beamWidth > 1) {
        val decoder = BeamSearchDecoder(
          cell, initialState, embeddingFn, tf.fill(INT32, tf.shape(srcSequenceLengths)(0).expandDims(0))(tgtBosID),
          tgtEosID, beamWidth, GooglePenalty(lengthPenaltyWeight), outputLayer)
        val tuple = decoder.decode(
          outputTimeMajor = timeMajor, maximumIterations = maxDecodingLength,
          parallelIterations = env.parallelIterations, swapMemory = env.swapMemory)
        RNNDecoder.Output(tuple._1.predictedIDs(---, 0), tuple._3(---, 0).cast(INT32))
      } else {
        val decHelper = BasicDecoder.GreedyEmbeddingHelper[DS](
          embeddingFn, tf.fill(INT32, tf.shape(srcSequenceLengths)(0).expandDims(0))(tgtBosID), tgtEosID)
        val decoder = BasicDecoder(cell, initialState, decHelper, outputLayer)
        val tuple = decoder.decode(
          outputTimeMajor = timeMajor, maximumIterations = maxDecodingLength,
          parallelIterations = env.parallelIterations, swapMemory = env.swapMemory)
        RNNDecoder.Output(tuple._1.sample, tuple._3)
      }
    }
  }
}

object RNNDecoder {
  case class Output(sequences: tf.Output, sequenceLengths: tf.Output)
}
