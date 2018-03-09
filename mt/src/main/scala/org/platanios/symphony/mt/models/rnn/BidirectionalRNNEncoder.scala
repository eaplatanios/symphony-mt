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
import org.platanios.symphony.mt.models.{DeviceManager, ParameterManager, RNNModel, Stage}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple

/**
  * @author Emmanouil Antonios Platanios
  */
class BidirectionalRNNEncoder[S, SS](
    val cell: Cell[S, SS],
    val numUnits: Int,
    val numLayers: Int,
    val dataType: DataType = FLOAT32,
    val residual: Boolean = false,
    val dropout: Option[Float] = None,
    val residualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output)
)(implicit
    evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) extends RNNEncoder[S, SS]()(evS, evSDropout) {
  override def create(
      config: RNNModel.Config[_, _],
      srcLanguage: Output,
      tgtLanguage: Output,
      srcSequences: Output,
      srcSequenceLengths: Output
  )(
      mode: Mode,
      env: Environment,
      parameterManager: ParameterManager,
      deviceManager: DeviceManager
  )(implicit
      stage: Stage
  ): Tuple[Output, Seq[S]] = {
    val transposedSequences = if (config.timeMajor) srcSequences.transpose() else srcSequences
    val embeddedSequences = parameterManager.wordEmbeddings(srcLanguage).gather(transposedSequences)
    val numResLayers = if (residual && numLayers > 1) numLayers - 1 else 0

    val biCellFw = RNNModel.multiCell(
      cell, embeddedSequences.shape(-1), numUnits, dataType, numLayers / 2, numResLayers / 2, dropout, residualFn,
      config.env.randomSeed, "MultiBiCellFw")(mode, env, parameterManager, deviceManager)
    val biCellBw = RNNModel.multiCell(
      cell, embeddedSequences.shape(-1), numUnits, dataType, numLayers / 2, numResLayers / 2, dropout, residualFn,
      config.env.randomSeed, "MultiBiCellBw")(mode, env, parameterManager, deviceManager)

    val unmergedBiTuple = tf.bidirectionalDynamicRNN(
      biCellFw, biCellBw, embeddedSequences, null, null, config.timeMajor,
      config.env.parallelIterations, config.env.swapMemory, srcSequenceLengths, "BidirectionalLayers")

    Tuple(
      tf.concatenate(Seq(unmergedBiTuple._1.output, unmergedBiTuple._2.output), -1),
      unmergedBiTuple._1.state.map(List(_))
          .zipAll(unmergedBiTuple._2.state.map(List(_)), Nil, Nil)
          .flatMap(Function.tupled(_ ::: _)))
  }
}

object BidirectionalRNNEncoder {
  def apply[S, SS](
      cell: Cell[S, SS],
      numUnits: Int,
      numLayers: Int,
      dataType: DataType = FLOAT32,
      residual: Boolean = false,
      dropout: Option[Float] = None,
      residualFn: Option[(Output, Output) => Output] = Some((input: Output, output: Output) => input + output)
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): BidirectionalRNNEncoder[S, SS] = {
    new BidirectionalRNNEncoder[S, SS](
      cell, numUnits, numLayers, dataType, residual, dropout, residualFn)(evS, evSDropout)
  }
}
