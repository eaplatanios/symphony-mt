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

import org.platanios.symphony.mt.{Environment, Language}
import org.platanios.symphony.mt.data.Vocabulary
import org.platanios.symphony.mt.models.StateBasedModel
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple

/**
  * @author Emmanouil Antonios Platanios
  */
class UnidirectionalRNNEncoder[S, SS](
    val srcLanguage: Language,
    val srcVocabulary: Vocabulary,
    val env: Environment,
    val cell: Cell[S, SS],
    val numUnits: Int,
    val numLayers: Int,
    val dataType: DataType = FLOAT32,
    val residual: Boolean = false,
    val dropout: Option[Float] = None,
    val residualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output),
    val timeMajor: Boolean = false
)(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) extends RNNEncoder[S, SS]()(evS, evSDropout) {
  override def create(inputSequences: Output, sequenceLengths: Output, mode: Mode): Tuple[Output, Seq[S]] = {
    // Time-major transpose
    val transposedSequences = if (timeMajor) inputSequences.transpose() else inputSequences

    // Embeddings
    val embeddings = StateBasedModel.embeddings(dataType, srcVocabulary.size, numUnits, "Embeddings")
    val embeddedSequences = tf.embeddingLookup(embeddings, transposedSequences)

    // RNN
    val numResLayers = if (residual && numLayers > 1) numLayers - 1 else 0
    val uniCell = StateBasedModel.multiCell(
      cell, numUnits, dataType, numLayers, numResLayers, dropout,
      residualFn, 0, env.numGPUs, env.randomSeed, "MultiUniCell")
    val createdCell = uniCell.createCellWithoutContext(mode, embeddedSequences.shape)
    tf.dynamicRNN(
      createdCell, embeddedSequences, null, timeMajor, env.parallelIterations, env.swapMemory, sequenceLengths,
      "UnidirectionalLayers")
  }
}

object UnidirectionalRNNEncoder {
  def apply[S, SS](
      srcLanguage: Language,
      srcVocabulary: Vocabulary,
      env: Environment,
      cell: Cell[S, SS],
      numUnits: Int,
      numLayers: Int,
      dataType: DataType = FLOAT32,
      residual: Boolean = false,
      dropout: Option[Float] = None,
      residualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output),
      timeMajor: Boolean = false
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): UnidirectionalRNNEncoder[S, SS] = {
    new UnidirectionalRNNEncoder[S, SS](
      srcLanguage, srcVocabulary, env, cell, numUnits, numLayers, dataType, residual, dropout, residualFn,
      timeMajor)(evS, evSDropout)
  }
}
