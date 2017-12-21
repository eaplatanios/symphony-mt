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

package org.platanios.symphony.mt.models.rnn

import org.platanios.symphony.mt.data.{DataConfig, Vocabulary}
import org.platanios.symphony.mt.models.InferConfig
import org.platanios.symphony.mt.{Environment, Language}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.{EVALUATION, INFERENCE, Mode}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple
import org.platanios.tensorflow.api.ops.seq2seq.decoders.{BasicDecoder, BeamSearchDecoder, GooglePenalty}

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Decoder[S, SS]()(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) {
  def create(
      encoderTuple: Tuple[Output, Seq[S]], inputSequenceLengths: Output,
      targetSequences: Output = null, targetSequenceLengths: Output = null, mode: Mode
  ): Decoder.Output
}

object Decoder {
  case class Output(sequences: tf.Output, sequenceLengths: tf.Output)
}

class UnidirectionalDecoder[S, SS](
    val tgtLanguage: Language,
    val tgtVocabulary: Vocabulary,
    val env: Environment,
    val rnnConfig: RNNConfig,
    val dataConfig: DataConfig,
    val inferConfig: InferConfig,
    val cell: Cell[S, SS],
    val numUnits: Int,
    val numLayers: Int,
    val dataType: DataType = FLOAT32,
    val residual: Boolean = false,
    val dropout: Option[Float] = None,
    val residualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output),
    val attention: Option[Attention] = None,
    val outputAttention: Boolean = false
)(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) extends Decoder[S, SS]()(evS, evSDropout) {
  override def create(
      encoderTuple: Tuple[Output, Seq[S]], inputSequenceLengths: Output,
      targetSequences: Output, targetSequenceLengths: Output, mode: Mode
  ): Decoder.Output = {
    // Embeddings
    val embeddings = Model.embeddings(dataType, tgtVocabulary.size, numUnits, "Embeddings")

    // Decoder RNN cell
    val numResLayers = if (residual && numLayers > 1) numLayers - 1 else 0
    val uniCell = Model.multiCell(
      cell, numUnits, dataType, numLayers, numResLayers, dropout,
      residualFn, 0, env.numGPUs, env.randomSeed, "MultiUniCell")

    // Use attention if necessary
    var initialState = encoderTuple.state
    var memory = {
      if (rnnConfig.timeMajor && mode.isTraining)
        encoderTuple.output.transpose(Tensor(1, 0, 2))
      else
        encoderTuple.output
    }
    var memorySequenceLengths = inputSequenceLengths
    if (inferConfig.beamWidth > 1) {
      // TODO: Find a way to remove the need for this tiling that is external to the beam search decoder.
      initialState = BeamSearchDecoder.tileForBeamSearch(initialState, inferConfig.beamWidth)
      memory = BeamSearchDecoder.tileForBeamSearch(memory, inferConfig.beamWidth)
      memorySequenceLengths = BeamSearchDecoder.tileForBeamSearch(memorySequenceLengths, inferConfig.beamWidth)
    }
    val (cellInstance, processedInitialState) = attention match {
      case None => (uniCell.createCell(mode, Shape(numUnits)), encoderTuple.state)
      case Some(a) => a.create(
        uniCell, memory, memorySequenceLengths, numUnits, numUnits, initialState, outputAttention, mode)
    }

    // Output layers
    val outputWeights = tf.variable(
      "OutWeights", dataType, Shape(cellInstance.cell.outputShape(-1), tgtVocabulary.size),
      tf.RandomUniformInitializer(-0.1f, 0.1f))
    val outputLayer = (logits: Output) => tf.linear(logits, outputWeights.value)

    if (mode.isTraining) {
      // Time-major transpose
      val transposedSequences = if (rnnConfig.timeMajor) targetSequences.transpose() else targetSequences
      val embeddedSequences = tf.embeddingLookup(embeddings, transposedSequences)

      // Decoder RNN
      val helper = BasicDecoder.TrainingHelper(embeddedSequences, targetSequenceLengths, rnnConfig.timeMajor)
      val decoder = BasicDecoder(cellInstance.cell, processedInitialState, helper, outputLayer)
      val tuple = decoder.decode(
        outputTimeMajor = rnnConfig.timeMajor, parallelIterations = rnnConfig.parallelIterations,
        swapMemory = rnnConfig.swapMemory)
      val lengths = tuple._3
      mode match {
        case INFERENCE | EVALUATION => Decoder.Output(tuple._1.sample, lengths)
        case _ => Decoder.Output(tuple._1.rnnOutput, lengths)
      }
    } else {
      // Decoder embeddings
      val embeddingFn = (o: Output) => tf.embeddingLookup(embeddings, o)
      val tgtVocabLookupTable = tgtVocabulary.lookupTable()
      val tgtBosID = tgtVocabLookupTable.lookup(tf.constant(dataConfig.beginOfSequenceToken)).cast(INT32)
      val tgtEosID = tgtVocabLookupTable.lookup(tf.constant(dataConfig.endOfSequenceToken)).cast(INT32)
      val maxDecodingLength = {
        if (dataConfig.tgtMaxLength != -1)
          tf.constant(dataConfig.tgtMaxLength)
        else
          tf.round(tf.max(tf.max(inputSequenceLengths)) * inferConfig.decoderMaxLengthFactor).cast(INT32)
      }

      // Decoder RNN
      if (inferConfig.beamWidth > 1) {
        val decoder = BeamSearchDecoder(
          cellInstance.cell, initialState, embeddingFn, tf.fill(INT32, tf.shape(inputSequenceLengths))(tgtBosID),
          tgtEosID, inferConfig.beamWidth, GooglePenalty(inferConfig.lengthPenaltyWeight), outputLayer)
        val tuple = decoder.decode(
          outputTimeMajor = rnnConfig.timeMajor, maximumIterations = maxDecodingLength,
          parallelIterations = rnnConfig.parallelIterations, swapMemory = rnnConfig.swapMemory)
        Decoder.Output(tuple._1.predictedIDs(---, 0), tuple._3(---, 0).cast(INT32))
      } else {
        val decHelper = BasicDecoder.GreedyEmbeddingHelper(
          embeddingFn, tf.fill(INT32, tf.shape(inputSequenceLengths))(tgtBosID), tgtEosID)
        val decoder = BasicDecoder(cellInstance.cell, initialState, decHelper, outputLayer)
        val tuple = decoder.decode(
          outputTimeMajor = rnnConfig.timeMajor, maximumIterations = maxDecodingLength,
          parallelIterations = rnnConfig.parallelIterations, swapMemory = rnnConfig.swapMemory)
        Decoder.Output(tuple._1.sample, tuple._3)
      }
    }
  }
}
