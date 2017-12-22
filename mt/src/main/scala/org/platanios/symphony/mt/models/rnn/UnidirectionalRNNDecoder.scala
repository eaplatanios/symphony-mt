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

import org.platanios.symphony.mt.{Environment, Language}
import org.platanios.symphony.mt.data.{DataConfig, Vocabulary}
import org.platanios.symphony.mt.models.InferConfig
import org.platanios.symphony.mt.models.attention.Attention
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.cell.{RNNCell, Tuple}
import org.platanios.tensorflow.api.ops.seq2seq.decoders.{BasicDecoder, BeamSearchDecoder, GooglePenalty}

/**
  * @author Emmanouil Antonios Platanios
  */
class UnidirectionalRNNDecoder[S, SS](
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
) extends RNNDecoder[S, SS]()(evS, evSDropout) {
  override def create(
      encoderTuple: Tuple[Output, Seq[S]], inputSequenceLengths: Output,
      targetSequences: Output, targetSequenceLengths: Output, mode: Mode
  ): RNNDecoder.Output = {
    // Embeddings
    val embeddings = Model.embeddings(dataType, tgtVocabulary.size, numUnits, "Embeddings")

    // Decoder RNN cell
    val numResLayers = if (residual && numLayers > 1) numLayers - 1 else 0
    val uniCell = Model.multiCell(
      cell, numUnits, dataType, numLayers, numResLayers, dropout,
      residualFn, 0, env.numGPUs, env.randomSeed, "MultiUniCell")

    // Use attention if necessary and create the decoder RNN
    var initialState = encoderTuple.state
    var memory = {
      if (rnnConfig.timeMajor)
        encoderTuple.output.transpose(Tensor(1, 0, 2))
      else
        encoderTuple.output
    }
    var memorySequenceLengths = inputSequenceLengths
    if (inferConfig.beamWidth > 1 && !mode.isTraining) {
      // TODO: Find a way to remove the need for this tiling that is external to the beam search decoder.
      initialState = BeamSearchDecoder.tileForBeamSearch(initialState, inferConfig.beamWidth)
      memory = BeamSearchDecoder.tileForBeamSearch(memory, inferConfig.beamWidth)
      memorySequenceLengths = BeamSearchDecoder.tileForBeamSearch(memorySequenceLengths, inferConfig.beamWidth)
    }

    attention match {
      case None =>
        decode(
          inputSequenceLengths, targetSequences, targetSequenceLengths, initialState,
          embeddings, uniCell.createCell(mode, Shape(numUnits)), mode)
      case Some(attentionCreator) =>
        val (attentionCell, attentionInitialState) = attentionCreator.create(
          uniCell, memory, memorySequenceLengths, numUnits, numUnits, initialState, outputAttention, mode)
        decode(
          inputSequenceLengths, targetSequences, targetSequenceLengths, attentionInitialState,
          embeddings, attentionCell, mode)
    }
  }

  protected def decode[DS, DSS](
      inputSequenceLengths: Output,
      targetSequences: Output,
      targetSequenceLengths: Output,
      initialState: DS,
      embeddings: Variable,
      cell: RNNCell[Output, Shape, DS, DSS],
      mode: Mode
  )(implicit
      evS: WhileLoopVariable.Aux[DS, DSS]
  ): RNNDecoder.Output = {
    val outputWeights = tf.variable(
      "OutWeights", embeddings.dataType, Shape(cell.outputShape(-1), tgtVocabulary.size),
      tf.RandomUniformInitializer(-0.1f, 0.1f))
    val outputLayer = (logits: Output) => tf.linear(logits, outputWeights.value)
    if (mode.isTraining) {
      // Time-major transpose
      val transposedSequences = if (rnnConfig.timeMajor) targetSequences.transpose() else targetSequences
      val embeddedSequences = tf.embeddingLookup(embeddings, transposedSequences)

      // Decoder RNN
      val helper = BasicDecoder.TrainingHelper(embeddedSequences, targetSequenceLengths, rnnConfig.timeMajor)
      val decoder = BasicDecoder(cell, initialState, helper, outputLayer)
      val tuple = decoder.decode(
        outputTimeMajor = rnnConfig.timeMajor, parallelIterations = rnnConfig.parallelIterations,
        swapMemory = rnnConfig.swapMemory)
      RNNDecoder.Output(tuple._1.rnnOutput, tuple._3)
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
          cell, initialState, embeddingFn, tf.fill(INT32, tf.shape(inputSequenceLengths)(0).expandDims(0))(tgtBosID),
          tgtEosID, inferConfig.beamWidth, GooglePenalty(inferConfig.lengthPenaltyWeight), outputLayer)
        val tuple = decoder.decode(
          outputTimeMajor = rnnConfig.timeMajor, maximumIterations = maxDecodingLength,
          parallelIterations = rnnConfig.parallelIterations, swapMemory = rnnConfig.swapMemory)
        RNNDecoder.Output(tuple._1.predictedIDs(---, 0), tuple._3(---, 0).cast(INT32))
      } else {
        val decHelper = BasicDecoder.GreedyEmbeddingHelper[DS](
          embeddingFn, tf.fill(INT32, tf.shape(inputSequenceLengths)(0).expandDims(0))(tgtBosID), tgtEosID)
        val decoder = BasicDecoder(cell, initialState, decHelper, outputLayer)
        val tuple = decoder.decode(
          outputTimeMajor = rnnConfig.timeMajor, maximumIterations = maxDecodingLength,
          parallelIterations = rnnConfig.parallelIterations, swapMemory = rnnConfig.swapMemory)
        RNNDecoder.Output(tuple._1.sample, tuple._3)
      }
    }
  }
}

object UnidirectionalRNNDecoder {
  def apply[S, SS](
      tgtLanguage: Language,
      tgtVocabulary: Vocabulary,
      env: Environment,
      rnnConfig: RNNConfig,
      dataConfig: DataConfig,
      inferConfig: InferConfig,
      cell: Cell[S, SS],
      numUnits: Int,
      numLayers: Int,
      dataType: DataType = FLOAT32,
      residual: Boolean = false,
      dropout: Option[Float] = None,
      residualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output),
      attention: Option[Attention] = None,
      outputAttention: Boolean = false
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): UnidirectionalRNNDecoder[S, SS] = {
    new UnidirectionalRNNDecoder[S, SS](
      tgtLanguage, tgtVocabulary, env, rnnConfig, dataConfig, inferConfig, cell, numUnits, numLayers, dataType,
      residual, dropout, residualFn, attention, outputAttention)(evS, evSDropout)
  }
}
