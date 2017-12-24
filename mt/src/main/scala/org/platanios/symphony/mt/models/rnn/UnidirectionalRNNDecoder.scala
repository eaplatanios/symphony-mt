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
import org.platanios.symphony.mt.models.{InferConfig, Model}
import org.platanios.symphony.mt.models.attention.Attention
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple
import org.platanios.tensorflow.api.ops.seq2seq.decoders.BeamSearchDecoder

/**
  * @author Emmanouil Antonios Platanios
  */
class UnidirectionalRNNDecoder[S, SS](
    override val tgtLanguage: Language,
    override val tgtVocabulary: Vocabulary,
    override val env: Environment,
    override val dataConfig: DataConfig,
    override val inferConfig: InferConfig,
    val cell: Cell[S, SS],
    val numUnits: Int,
    val numLayers: Int,
    val dataType: DataType = FLOAT32,
    val residual: Boolean = false,
    val dropout: Option[Float] = None,
    val residualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output),
    val attention: Option[Attention] = None,
    val outputAttention: Boolean = true,
    override val timeMajor: Boolean = false
)(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) extends RNNDecoder[S, SS](tgtLanguage, tgtVocabulary, env, dataConfig, inferConfig, timeMajor)(evS, evSDropout) {
  override def create(
      encoderTuple: Tuple[Output, Seq[S]], inputSequenceLengths: Output,
      targetSequences: Output, targetSequenceLengths: Output, mode: Mode
  ): RNNDecoder.Output = {
    // Embeddings
    val embeddings = Model.embeddings(dataType, tgtVocabulary.size, numUnits, "Embeddings")

    // RNN cell
    val numResLayers = if (residual && numLayers > 1) numLayers - 1 else 0
    val uniCell = Model.multiCell(
      cell, numUnits, dataType, numLayers, numResLayers, dropout,
      residualFn, 0, env.numGPUs, env.randomSeed, "MultiUniCell")

    // Use attention if necessary and create the decoder RNN
    var initialState = encoderTuple.state
    var memory = if (timeMajor) encoderTuple.output.transpose(Tensor(1, 0, 2)) else encoderTuple.output
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
          uniCell, memory, memorySequenceLengths, numUnits, numUnits, initialState, useAttentionLayer = true,
          outputAttention = outputAttention, mode)
        decode(
          inputSequenceLengths, targetSequences, targetSequenceLengths, attentionInitialState,
          embeddings, attentionCell, mode)
    }
  }
}

object UnidirectionalRNNDecoder {
  def apply[S, SS](
      tgtLanguage: Language,
      tgtVocabulary: Vocabulary,
      env: Environment,
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
      outputAttention: Boolean = false,
      timeMajor: Boolean = false
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): UnidirectionalRNNDecoder[S, SS] = {
    new UnidirectionalRNNDecoder[S, SS](
      tgtLanguage, tgtVocabulary, env, dataConfig, inferConfig, cell, numUnits, numLayers, dataType, residual, dropout,
      residualFn, attention, outputAttention, timeMajor)(evS, evSDropout)
  }
}
