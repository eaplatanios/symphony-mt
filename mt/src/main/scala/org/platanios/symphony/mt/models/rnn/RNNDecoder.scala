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
import org.platanios.symphony.mt.models._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple
import org.platanios.tensorflow.api.ops.seq2seq.decoders.{BasicDecoder, BeamSearchDecoder, GooglePenalty}

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class RNNDecoder[S, SS]()(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) extends Decoder[(Tuple[Output, Seq[S]], Output, Output)] {
  def create(
      config: RNNModel.Config[_, _],
      srcLanguage: Output,
      tgtLanguage: Output,
      encoderState: (Tuple[Output, Seq[S]], Output, Output),
      beginOfSequenceToken: String,
      endOfSequenceToken: String,
      tgtSequences: Output = null,
      tgtSequenceLengths: Output = null
  )(implicit
      stage: Stage,
      mode: Mode,
      env: Environment,
      parameterManager: ParameterManager,
      deviceManager: DeviceManager
  ): RNNDecoder.Output

  protected def decode[DS, DSS](
      config: RNNModel.Config[_, _],
      srcSequenceLengths: Output,
      tgtLanguage: Output,
      tgtSequences: Output,
      tgtSequenceLengths: Output,
      initialState: DS,
      embeddings: Output => Output,
      cell: tf.RNNCell[Output, Shape, DS, DSS],
      tgtMaxLength: Output,
      beginOfSequenceToken: String,
      endOfSequenceToken: String
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      evS: WhileLoopVariable.Aux[DS, DSS]
  ): RNNDecoder.Output = {
    val outputWeights = parameterManager.getProjectionToWords(cell.outputShape(-1), tgtLanguage)
    val outputLayer = (logits: Output) => tf.linear(logits, outputWeights)
    if (mode.isTraining) {
      // Time-major transpose
      val transposedSequences = if (config.timeMajor) tgtSequences.transpose() else tgtSequences
      val embeddedSequences = embeddings(transposedSequences)

      // Decoder RNN
      val helper = BasicDecoder.TrainingHelper(embeddedSequences, tgtSequenceLengths, config.timeMajor)
      val decoder = BasicDecoder(cell, initialState, helper, outputLayer)
      val tuple = decoder.decode(
        outputTimeMajor = config.timeMajor, parallelIterations = config.env.parallelIterations,
        swapMemory = config.env.swapMemory)
      RNNDecoder.Output(tuple._1.rnnOutput, tuple._3)
    } else {
      // Decoder embeddings
      val embeddingFn = (o: Output) => embeddings(o)
      val tgtVocabLookupTable = parameterManager.stringToIndexLookup(tgtLanguage)
      val tgtBosID = tgtVocabLookupTable(tf.constant(beginOfSequenceToken)).cast(INT32)
      val tgtEosID = tgtVocabLookupTable(tf.constant(endOfSequenceToken)).cast(INT32)

      // Decoder RNN
      if (config.beamWidth > 1) {
        val decoder = BeamSearchDecoder(
          cell, initialState, embeddingFn, tf.fill(INT32, tf.shape(srcSequenceLengths)(0).expandDims(0))(tgtBosID),
          tgtEosID, config.beamWidth, GooglePenalty(config.lengthPenaltyWeight), outputLayer)
        val tuple = decoder.decode(
          outputTimeMajor = config.timeMajor, maximumIterations = tgtMaxLength,
          parallelIterations = config.env.parallelIterations, swapMemory = config.env.swapMemory)
        RNNDecoder.Output(tuple._1.predictedIDs(---, 0), tuple._3(---, 0).cast(INT32))
      } else {
        val decHelper = BasicDecoder.GreedyEmbeddingHelper[DS](
          embeddingFn, tf.fill(INT32, tf.shape(srcSequenceLengths)(0).expandDims(0))(tgtBosID), tgtEosID)
        val decoder = BasicDecoder(cell, initialState, decHelper, outputLayer)
        val tuple = decoder.decode(
          outputTimeMajor = config.timeMajor, maximumIterations = tgtMaxLength,
          parallelIterations = config.env.parallelIterations, swapMemory = config.env.swapMemory)
        RNNDecoder.Output(tuple._1.sample, tuple._3)
      }
    }
  }
}

object RNNDecoder {
  case class Output(sequences: tf.Output, sequenceLengths: tf.Output)
}
