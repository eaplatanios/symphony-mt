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

import org.platanios.symphony.mt.data.Vocabulary
import org.platanios.symphony.mt.models.StateBasedModel
import org.platanios.symphony.mt.{Environment, Language}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple

/**
  * @author Emmanouil Antonios Platanios
  */
class GNMTEncoder[S, SS](
    val srcLanguage: Language,
    val srcVocabulary: Vocabulary,
    val env: Environment,
    val cell: Cell[S, SS],
    val numUnits: Int,
    val numBiLayers: Int,
    val numUniLayers: Int,
    val numUniResLayers: Int,
    val dataType: DataType = FLOAT32,
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

    // Bidirectional RNN layers
    val biTuple = {
      if (numBiLayers > 0) {
        val biCellFw = StateBasedModel.multiCell(
          cell, numUnits, dataType, numBiLayers, 0, dropout,
          residualFn, 0, env.numGPUs, env.randomSeed, "MultiBiCellFw")
        val biCellBw = StateBasedModel.multiCell(
          cell, numUnits, dataType, numBiLayers, 0, dropout,
          residualFn, numBiLayers, env.numGPUs, env.randomSeed, "MultiBiCellBw")
        val createdCellFw = biCellFw.createCell(mode, embeddedSequences.shape)
        val createdCellBw = biCellBw.createCell(mode, embeddedSequences.shape)
        val unmergedBiTuple = tf.bidirectionalDynamicRNN(
          createdCellFw, createdCellBw, embeddedSequences, null, null, timeMajor, env.parallelIterations,
          env.swapMemory, sequenceLengths, "BidirectionalLayers")
        Tuple(
          tf.concatenate(Seq(unmergedBiTuple._1.output, unmergedBiTuple._2.output), -1), unmergedBiTuple._2.state)
      } else {
        Tuple(embeddedSequences, Seq.empty[S])
      }
    }

    // Unidirectional RNN layers
    val uniCell = StateBasedModel.multiCell(
      cell, numUnits, dataType, numUniLayers, numUniResLayers, dropout, residualFn,
      2 * numBiLayers, env.numGPUs, env.randomSeed, "MultiUniCell")
    val uniCellInstance = uniCell.createCell(mode, biTuple.output.shape)
    val uniTuple = tf.dynamicRNN(
      uniCellInstance, biTuple.output, null, timeMajor, env.parallelIterations, env.swapMemory, sequenceLengths,
      "UnidirectionalLayers")

    // Pass all of the encoder's state except for the first bi-directional layer's state, to the decoder.
    Tuple(uniTuple.output, biTuple.state ++ uniTuple.state)
  }
}

object GNMTEncoder {
  def apply[S, SS](
      srcLanguage: Language,
      srcVocabulary: Vocabulary,
      env: Environment,
      cell: Cell[S, SS],
      numUnits: Int,
      numBiLayers: Int,
      numUniLayers: Int,
      numUniResLayers: Int,
      dataType: DataType = FLOAT32,
      dropout: Option[Float] = None,
      residualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output),
      timeMajor: Boolean = false
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): GNMTEncoder[S, SS] = {
    new GNMTEncoder[S, SS](
      srcLanguage, srcVocabulary, env, cell, numUnits, numBiLayers, numUniLayers, numUniResLayers, dataType, dropout,
      residualFn, timeMajor)(evS, evSDropout)
  }
}
