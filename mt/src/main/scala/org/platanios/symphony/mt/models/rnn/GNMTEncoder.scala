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
import org.platanios.symphony.mt.models.{ParametersManager, RNNModel}
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple

/**
  * @author Emmanouil Antonios Platanios
  */
class GNMTEncoder[S, SS](
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
  override def create[I](
      env: Environment,
      srcSequences: Output,
      srcSequenceLengths: Output,
      srcVocab: Vocabulary
  )(mode: Mode, parametersManager: ParametersManager[I]): Tuple[Output, Seq[S]] = {
    // Time-major transpose
    val transposedSequences = if (timeMajor) srcSequences.transpose() else srcSequences

    // Embeddings
    val embeddings = RNNModel.embeddings(dataType, srcVocab.size, numUnits, "Embeddings")
    val embeddedSequences = tf.embeddingLookup(embeddings, transposedSequences)

    // Bidirectional RNN layers
    val biTuple = {
      if (numBiLayers > 0) {
        val biCellFw = RNNModel.multiCell(
          cell, embeddedSequences.shape(-1), numUnits, dataType, numBiLayers, 0, dropout,
          residualFn, 0, env.numGPUs, env.firstGPU, env.randomSeed, "MultiBiCellFw")(mode, parametersManager)
        val biCellBw = RNNModel.multiCell(
          cell, embeddedSequences.shape(-1), numUnits, dataType, numBiLayers, 0, dropout,
          residualFn, numBiLayers, env.numGPUs, env.firstGPU, env.randomSeed, "MultiBiCellBw")(mode, parametersManager)
        val unmergedBiTuple = tf.bidirectionalDynamicRNN(
          biCellFw, biCellBw, embeddedSequences, null, null, timeMajor, env.parallelIterations,
          env.swapMemory, srcSequenceLengths, "BidirectionalLayers")
        Tuple(
          tf.concatenate(Seq(unmergedBiTuple._1.output, unmergedBiTuple._2.output), -1), unmergedBiTuple._2.state)
      } else {
        Tuple(embeddedSequences, Seq.empty[S])
      }
    }

    // Unidirectional RNN layers
    val uniCell = RNNModel.multiCell(
      cell, biTuple.output.shape(-1), numUnits, dataType, numUniLayers, numUniResLayers, dropout, residualFn,
      2 * numBiLayers, env.numGPUs, env.firstGPU, env.randomSeed, "MultiUniCell")(mode, parametersManager)
    val uniTuple = tf.dynamicRNN(
      uniCell, biTuple.output, null, timeMajor, env.parallelIterations, env.swapMemory, srcSequenceLengths,
      "UnidirectionalLayers")

    // Pass all of the encoder's state except for the first bi-directional layer's state, to the decoder.
    Tuple(uniTuple.output, biTuple.state ++ uniTuple.state)
  }
}

object GNMTEncoder {
  def apply[S, SS](
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
      cell, numUnits, numBiLayers, numUniLayers, numUniResLayers, dataType, dropout,
      residualFn, timeMajor)(evS, evSDropout)
  }
}
