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

package org.platanios.symphony.mt.models.helpers

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.training.optimizers.schedules.{FixedSchedule, Schedule}
import org.platanios.tensorflow.api.ops.variables.ZerosInitializer

/** Optimizer that implements the AMSGrad optimization algorithm, presented in
  * [On the Convergence of Adam and Beyond](https://openreview.net/pdf?id=ryQu7f-RZ).
  *
  * Initialization:
  * {{{
  *   m_0 = 0     // Initialize the 1st moment vector
  *   v_0 = 0     // Initialize the 2nd moment vector
  *   v_hat_0 = 0 // Initialize the 2nd moment max vector
  *   t = 0       // Initialize the time step
  * }}}
  *
  * The AMSGrad update for step `t` is as follows:
  * {{{
  *   learningRate_t = initialLearningRate * sqrt(beta1 - beta2^t) / (1 - beta1^t)
  *   m_t = beta1 * m_{t-1} + (1 - beta1) * gradient
  *   v_t = beta2 * v_{t-1} + (1 - beta2) * gradient * gradient
  *   v_hat_t = max(v_t, v_hat_{t-1})
  *   variable -= learningRate_t * m_t / (sqrt(v_hat_t) + epsilon)
  * }}}
  *
  * The default value of `1e-8` for epsilon might not be a good default in general. For example, when training an
  * Inception network on ImageNet a current good choice is `1.0` or `0.1`.
  *
  * The sparse implementation of this algorithm (used when the gradient is an indexed slices object, typically because
  * of `tf.gather` or an embedding lookup in the forward pass) does apply momentum to variable slices even if they were
  * not used in the forward pass (meaning they have a gradient equal to zero). Momentum decay (`beta1`) is also applied
  * to the entire momentum accumulator. This means that the sparse behavior is equivalent to the dense behavior (in
  * contrast to some momentum implementations which ignore momentum unless a variable slice was actually used).
  *
  * For more information on this algorithm, please refer to this [paper](https://openreview.net/pdf?id=ryQu7f-RZ).
  *
  * @param  learningRate           Learning rate. Must be `> 0`. If used with `decay`, then this argument
  *                                specifies the initial value of the learning rate.
  * @param  decay                  Learning rate decay method to use for each update.
  * @param  beta1                  Exponential decay rate for the first moment estimates.
  * @param  beta2                  Exponential decay rate for the second moment estimates.
  * @param  epsilon                Small constant used for numerical stability. This epsilon corresponds to
  *                                "epsilon hat" in the Kingma and Ba paper (in the formula just before Section 2.1),
  *                                and not to the epsilon in Algorithm 1 of the paper.
  * @param  useLocking             If `true`, the gradient descent updates will be protected by a lock. Otherwise, the
  *                                behavior is undefined, but may exhibit less contention.
  * @param  learningRateSummaryTag Optional summary tag name to use for the learning rate value. If `null`, no summary
  *                                is created for the learning rate. Otherwise, a scalar summary is created which can
  *                                be monitored using TensorBoard.
  * @param  name                   Name for this optimizer.
  *
  * @author Emmanouil Antonios Platanios
  */
class AdaptiveAMSGrad protected (
    val learningRate: Float = 0.001f,
    val decay: Schedule[Float] = FixedSchedule[Float](),
    val beta1: Float = 0.9f,
    val beta2: Float = 0.999f,
    val epsilon: Float = 1e-8f,
    val useLocking: Boolean = false,
    val learningRateSummaryTag: String = null,
    val name: String = "AMSGrad"
) extends tf.train.Optimizer {
  override val ignoreDuplicateSparseIndices: Boolean = true

  protected var learningRateTensor: Output[Float] = _
  protected var beta1Tensor       : Output[Float] = _
  protected var beta2Tensor       : Output[Float] = _
  protected var epsilonTensor     : Output[Float] = _

  protected def getLearningRate[V: TF, I: TF : IsIntOrLong](
      variable: Variable[V],
      iteration: Option[Variable[I]]
  ): Output[V] = {
    if (learningRateTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    learningRateTensor.castTo[V].toOutput
  }

  protected def getBeta1[V: TF](
      variable: Variable[V]
  ): Output[V] = {
    if (beta1Tensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    beta1Tensor.castTo[V].toOutput
  }

  protected def getBeta2[V: TF](
      variable: Variable[V]
  ): Output[V] = {
    if (beta2Tensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    beta2Tensor.castTo[V].toOutput
  }

  protected def getEpsilon[V: TF](
      variable: Variable[V]
  ): Output[V] = {
    if (epsilonTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    epsilonTensor.castTo[V].toOutput
  }

  override def createSlots(variables: Seq[Variable[Any]]): Unit = {
    // Create slots for the first and second moments.
    variables.foreach(v => {
      zerosSlot("M", v, name)(TF.fromDataType(v.dataType))
      zerosSlot("V", v, name)(TF.fromDataType(v.dataType))
      zerosSlot("Vhat", v, name)(TF.fromDataType(v.dataType))
      getSlot("NonZerosCount", v, INT64, ZerosInitializer, Shape(v.shape(0)), name)
      getSlot("Beta1Power", v, FLOAT32, ZerosInitializer, Shape(v.shape(0)), name)
      getSlot("Beta2Power", v, FLOAT32, ZerosInitializer, Shape(v.shape(0)), name)
    })
  }

  override def prepare[I: TF : IsIntOrLong](
      iteration: Option[Variable[I]]
  ): Unit = {
    learningRateTensor = decay(tf.constant(learningRate, name = "LearningRate"), iteration)
    if (learningRateSummaryTag != null)
      tf.summary.scalar(learningRateSummaryTag, learningRateTensor)
    beta1Tensor = tf.constant(beta1, name = "Beta1")
    beta2Tensor = tf.constant(beta2, name = "Beta2")
    epsilonTensor = tf.constant(epsilon, name = "Epsilon")
  }

  override def applyDense[T: TF : IsNotQuantized, I: TF : IsIntOrLong](
      gradient: Output[T],
      variable: Variable[T],
      iteration: Option[Variable[I]]
  ): UntypedOp = {
    tf.nameScope(s"$name/ApplyDense") {
      val m = getSlot[T, T]("M", variable)
      val v = getSlot[T, T]("V", variable)
      val vHat = getSlot[T, T]("Vhat", variable)
      val beta1Power = getSlot[T, Float]("Beta1Power", variable)
      val beta2Power = getSlot[T, Float]("Beta2Power", variable)

      val betaShape = Shape(variable.shape(0)) ++ Shape(Array.fill(variable.rank - 1)(1))

//      val nonZerosCount = getSlot[T, Long]("NonZerosCount", variable)
//      val nonZerosCountValue = nonZerosCount.assignAdd(tf.notEqual(gradient, tf.zeros[T](Shape())).toLong)

//      val initialBeta1 = getBeta1(variable).toFloat
//      val initialBeta2 = getBeta2(variable).toFloat
//      val aBeta1 = 1.0f - initialBeta1
//      val aBeta2 = 1.0f - initialBeta2
//      val bBeta1 = initialBeta1
//      val bBeta2 = initialBeta2
//      val rate = 1.0f - nonZerosCountValue.toFloat / (iteration.get.toFloat + 1.0f)
//      val beta1 = (aBeta1 * rate + bBeta1).castTo[T]
//      val beta2 = (aBeta2 * rate + bBeta2).castTo[T]

      val beta1 = getBeta1(variable)
      val beta2 = getBeta2(variable)

      val epsilon = getEpsilon(variable)

      var learningRate = getLearningRate(variable, iteration)
      val one = tf.ones[T](Shape())
      learningRate = learningRate * tf.sqrt(one - beta2Power.value.reshape(betaShape).castTo[T])
      learningRate = learningRate / (one - beta1Power.value.reshape(betaShape).castTo[T])

      // m_t = beta1 * m + (1 - beta1) * gradient
      val mScaledGradient = gradient * (one - beta1)
      val mT = m.assign((m.value * beta1) + mScaledGradient)

      // v_t = beta2 * v + (1 - beta2) * gradient * gradient
      val vScaledGradient = tf.square(gradient) * (one - beta2)
      val vT = v.assign((v.value * beta2) + vScaledGradient)

      val vHatT = vHat.assign(tf.maximum(vT, vHat))
      val vHatTSqrt = tf.sqrt(vHatT)
      val denominator = vHatTSqrt + epsilon
      val update = variable.assignSub(learningRate * mT / denominator)

      val updateBeta1Power = beta1Power.assign(beta1Power.value * beta1.toFloat)
      val updateBeta2Power = beta2Power.assign(beta2Power.value * beta2.toFloat)

      tf.group(Set(update.op, mT.op, vT.op, updateBeta1Power.op, updateBeta2Power.op))
    }
  }

  override def applySparse[T: TF : IsNotQuantized, I: TF : IsIntOrLong](
      gradient: OutputIndexedSlices[T],
      variable: Variable[T],
      iteration: Option[Variable[I]]
  ): UntypedOp = {
    tf.nameScope(s"$name/ApplySparse") {
      val m = getSlot[T, T]("M", variable)
      val v = getSlot[T, T]("V", variable)
      val vHat = getSlot[T, T]("Vhat", variable)
      val beta1Power = getSlot[T, Float]("Beta1Power", variable)
      val beta2Power = getSlot[T, Float]("Beta2Power", variable)

      val nonZerosCount = getSlot[T, Long]("NonZerosCount", variable)
      val nonZerosCountValue = nonZerosCount.assignScatterAdd(gradient.indices, 1L).toFloat
      // nonZerosCountValue = tf.print(nonZerosCountValue, Seq(nonZerosCountValue), s"Sparse ${variable.name}: ", 10000, 10000)

      val betaShape = Shape(variable.shape(0)) ++ Shape(Array.fill(variable.rank - 1)(1))

      val initialBeta1 = getBeta1(variable).toFloat
      val initialBeta2 = getBeta2(variable).toFloat
      val aBeta1 = 1.0f - initialBeta1
      val aBeta2 = 1.0f - initialBeta2
      val bBeta1 = initialBeta1
      val bBeta2 = initialBeta2
      val rate = 1.0f - nonZerosCountValue.toFloat / (iteration.get.toFloat + 1.0f)
      val beta1 = (aBeta1 * rate + bBeta1).castTo[T]
      val beta2 = (aBeta2 * rate + bBeta2).castTo[T]

      val epsilon = getEpsilon(variable)

      var learningRate = getLearningRate(variable, iteration)
      val one = tf.ones[T](Shape())
      learningRate = learningRate * tf.sqrt(one - beta2Power.value.reshape(betaShape).castTo[T])
      learningRate = learningRate / (one - beta1Power.value.reshape(betaShape).castTo[T])

      // m_t = beta1 * m + (1 - beta1) * gradient
      val mScaledGradient = gradient.values * (one - beta1.reshape(betaShape).gather(gradient.indices))
      var mT = m.assign(m.value * beta1.reshape(betaShape))
      mT = tf.createWith(controlDependencies = Set(mT.op)) {
        m.assignScatterAdd(gradient.indices, mScaledGradient)
      }

      // v_t = beta2 * v + (1 - beta2) * gradient * gradient
      val vScaledGradient = gradient.values * gradient.values * (one - beta2.reshape(betaShape).gather(gradient.indices))
      var vT = v.assign(v.value * beta2.reshape(betaShape))
      vT = tf.createWith(controlDependencies = Set(vT.op)) {
        v.assignScatterAdd(gradient.indices, vScaledGradient)
      }

      val vHatT = vHat.assign(tf.maximum(vT, vHat))
      val vHatTSqrt = tf.sqrt(vHatT)
      val denominator = vHatTSqrt + epsilon
      val update = variable.assignSub(learningRate * mT / denominator)

      val updateBeta1Power = beta1Power.assign(beta1Power.value * beta1.toFloat)
      val updateBeta2Power = beta2Power.assign(beta2Power.value * beta2.toFloat)

      tf.group(Set(update.op, mT.op, vT.op, updateBeta1Power.op, updateBeta2Power.op))
    }
  }
}

object AdaptiveAMSGrad {
  def apply(
      learningRate: Float = 0.001f,
      decay: Schedule[Float] = FixedSchedule[Float](),
      beta1: Float = 0.9f,
      beta2: Float = 0.999f,
      epsilon: Float = 1e-8f,
      useLocking: Boolean = false,
      learningRateSummaryTag: String = null,
      name: String = "AdaptiveAMSGrad"
  ): AdaptiveAMSGrad = {
    new AdaptiveAMSGrad(learningRate, decay, beta1, beta2, epsilon, useLocking, learningRateSummaryTag, name)
  }
}
