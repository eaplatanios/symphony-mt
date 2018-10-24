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

import org.platanios.symphony.mt.models.Stage
import org.platanios.symphony.mt.models.parameters.ParameterManager
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape, Zero}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops.rnn.cell._

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Cell[T: TF, State, StateShape](implicit
    val evOutputStructureState: OutputStructure[State],
    val evOutputToShapeState: OutputToShape.Aux[State, StateShape],
    val evZeroState: Zero.Aux[State, StateShape]
) {
  // The following two type aliases are used in the experiments module.
  type DataType = T
  type StateType = State
  type StateShapeType = StateShape

  def create(
      name: String,
      numInputs: Int,
      numUnits: Int
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      stage: Stage,
      context: Output[Int]
  ): RNNCell[Output[T], State, Shape, StateShape]
}

case class GRU[T: TF : IsNotQuantized](
    activation: Output[T] => Output[T]
) extends Cell[T, Output[T], Shape] {
  override def create(
      name: String,
      numInputs: Int,
      numUnits: Int
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      stage: Stage,
      context: Output[Int]
  ): RNNCell[Output[T], Output[T], Shape, Shape] = {
    val gateKernel = parameterManager.get[T]("Gate/Weights", Shape(numInputs + numUnits, 2 * numUnits))
    val gateBias = parameterManager.get[T]("Gate/Bias", Shape(2 * numUnits), tf.ZerosInitializer)
    val candidateKernel = parameterManager.get[T]("Candidate/Weights", Shape(numInputs + numUnits, numUnits))
    val candidateBias = parameterManager.get[T]("Candidate/Bias", Shape(numUnits), tf.ZerosInitializer)
    GRUCell(gateKernel, gateBias, candidateKernel, candidateBias, activation, name)
  }
}

case class BasicLSTM[T: TF : IsNotQuantized](
    activation: Output[T] => Output[T],
    forgetBias: Float = 1.0f
) extends Cell[T, LSTMState[T], (Shape, Shape)] {
  override def create(
      name: String,
      numInputs: Int,
      numUnits: Int
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      stage: Stage,
      context: Output[Int]
  ): BasicLSTMCell[T] = {
    val kernel = parameterManager.get[T]("Weights", Shape(numInputs + numUnits, 4 * numUnits))
    val bias = parameterManager.get[T]("Bias", Shape(4 * numUnits), tf.ZerosInitializer)
    BasicLSTMCell(kernel, bias, activation, forgetBias, name)
  }
}

case class LSTM[T: TF : IsNotQuantized](
    activation: Output[T] => Output[T],
    forgetBias: Float = 1.0f,
    usePeepholes: Boolean = false,
    cellClip: Float = -1,
    projectionSize: Int = -1,
    projectionClip: Float = -1
) extends Cell[T, LSTMState[T], (Shape, Shape)] {
  override def create(
      name: String,
      numInputs: Int,
      numUnits: Int
  )(implicit
      mode: Mode,
      parameterManager: ParameterManager,
      stage: Stage,
      context: Output[Int]
  ): LSTMCell[T] = {
    val hiddenDepth = if (projectionSize != -1) projectionSize else numUnits
    val kernel = parameterManager.get[T]("Weights", Shape(numInputs + hiddenDepth, 4 * numUnits))
    val bias = parameterManager.get[T]("Bias", Shape(4 * numUnits), tf.ZerosInitializer)
    val (wfDiag, wiDiag, woDiag) = {
      if (usePeepholes) {
        val wfDiag = parameterManager.get[T]("Peepholes/ForgetKernelDiag", Shape(numUnits))
        val wiDiag = parameterManager.get[T]("Peepholes/InputKernelDiag", Shape(numUnits))
        val woDiag = parameterManager.get[T]("Peepholes/OutputKernelDiag", Shape(numUnits))
        (wfDiag, wiDiag, woDiag)
      } else {
        (null, null, null)
      }
    }
    val projectionKernel = {
      if (projectionSize != -1)
        parameterManager.get[T]("Projection/Weights", Shape(numUnits, projectionSize))
      else
        null
    }
    LSTMCell(
      kernel, bias, activation, cellClip, wfDiag, wiDiag, woDiag,
      projectionKernel, projectionClip, forgetBias, name)
  }
}
