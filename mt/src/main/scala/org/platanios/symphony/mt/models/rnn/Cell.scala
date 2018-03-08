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

import org.platanios.symphony.mt.models.{ParameterManager, Stage}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.rnn.cell._
import org.platanios.tensorflow.api.types.DataType

/**
  * @author Emmanouil Antonios Platanios
  */
trait Cell[S, SS] {
  def create(
      name: String,
      numInputs: Int,
      numUnits: Int,
      dataType: DataType
  )(mode: Mode, parameterManager: ParameterManager)(implicit
      stage: Stage
  ): RNNCell[Output, Shape, S, SS]
}

case class GRU(activation: Output => Output = tf.tanh(_)) extends Cell[Output, Shape] {
  override def create(
      name: String,
      numInputs: Int,
      numUnits: Int,
      dataType: DataType
  )(mode: Mode, parameterManager: ParameterManager)(implicit
      stage: Stage
  ): RNNCell[Output, Shape, Output, Shape] = {
    val gateKernel = parameterManager.get("Gate/Weights", dataType, Shape(numInputs + numUnits, 2 * numUnits))
    val gateBias = parameterManager.get("Gate/Bias", dataType, Shape(2 * numUnits), tf.ZerosInitializer)
    val candidateKernel = parameterManager.get("Candidate/Weights", dataType, Shape(numInputs + numUnits, numUnits))
    val candidateBias = parameterManager.get("Candidate/Bias", dataType, Shape(numUnits), tf.ZerosInitializer)
    GRUCell(gateKernel, gateBias, candidateKernel, candidateBias, activation, name)
  }
}

case class BasicLSTM(forgetBias: Float = 1.0f, activation: Output => Output = tf.tanh(_))
    extends Cell[LSTMState, (Shape, Shape)] {
  override def create(
      name: String,
      numInputs: Int,
      numUnits: Int,
      dataType: DataType
  )(mode: Mode, parameterManager: ParameterManager)(implicit
      stage: Stage
  ): BasicLSTMCell = {
    val kernel = parameterManager.get("Weights", dataType, Shape(numInputs + numUnits, 4 * numUnits))
    val bias = parameterManager.get("Bias", dataType, Shape(4 * numUnits), tf.ZerosInitializer)
    BasicLSTMCell(kernel, bias, activation, forgetBias, name)
  }
}

case class LSTM(
    forgetBias: Float = 1.0f,
    usePeepholes: Boolean = false,
    cellClip: Float = -1,
    projectionSize: Int = -1,
    projectionClip: Float = -1,
    activation: Output => Output = tf.tanh(_)
) extends Cell[LSTMState, (Shape, Shape)] {
  override def create(
      name: String,
      numInputs: Int,
      numUnits: Int,
      dataType: DataType
  )(mode: Mode, parameterManager: ParameterManager)(implicit
      stage: Stage
  ): LSTMCell = {
    val hiddenDepth = if (projectionSize != -1) projectionSize else numUnits
    val kernel = parameterManager.get("Weights", dataType, Shape(numInputs + hiddenDepth, 4 * numUnits))
    val bias = parameterManager.get("Bias", dataType, Shape(4 * numUnits), tf.ZerosInitializer)
    val (wfDiag, wiDiag, woDiag) = {
      if (usePeepholes) {
        val wfDiag = parameterManager.get("Peepholes/ForgetKernelDiag", dataType, Shape(numUnits))
        val wiDiag = parameterManager.get("Peepholes/InputKernelDiag", dataType, Shape(numUnits))
        val woDiag = parameterManager.get("Peepholes/OutputKernelDiag", dataType, Shape(numUnits))
        (wfDiag, wiDiag, woDiag)
      } else {
        (null, null, null)
      }
    }
    val projectionKernel = {
      if (projectionSize != -1)
        parameterManager.get("Projection/Weights", dataType, Shape(numUnits, projectionSize))
      else
        null
    }
    LSTMCell(
      kernel, bias, cellClip, wfDiag, wiDiag, woDiag, projectionKernel, projectionClip, activation, forgetBias, name)
  }
}
