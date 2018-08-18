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

import org.platanios.symphony.mt.models.Stage
import org.platanios.symphony.mt.models.parameters.ParameterManager
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode

import scala.language.postfixOps

/**
  * @author Emmanouil Antonios Platanios
  */
object Common {
  /** Returns `true` if the current op creation context places ops on a TPU. */
  def isOnTPU: Boolean = {
    tf.currentNameScope.startsWith("TPUReplicate")
  }

  /** Reshapes `input` so that the last dimension becomes two dimensions.
    *
    * The first of these two dimensions will have size `n`.
    *
    * @param  input Tensor with shape `[..., m]`.
    * @param  n     Size at which to split the last dimension.
    * @return Tensor with shape `[..., n, m / n]`.
    */
  def splitLastDimension(input: Output, n: Int): Output = {
    val inputShape = tf.shape(input)
    val result = tf.reshape(input, tf.concatenate(Seq(
      inputShape(0 :: -1),
      tf.constant(n, shape = Shape(1)),
      tf.truncateDivide(inputShape(-1), n).expandDims(0))))
    if (input.rank != -1 && input.shape(-1) != -1)
      result.setShape(Shape.fromSeq(result.shape.asArray.updated(result.rank - 1, input.shape(-1) / n)))
    result
  }

  /** Reshapes `input` so that the last two dimensions become one.
    *
    * @param  input Tensor with shape `[..., a, b]`.
    * @return Tensor with shape `[..., a * b]`.
    */
  def combineLastTwoDimensions(input: Output): Output = {
    val inputShape = tf.shape(input)
    val result = tf.reshape(input, tf.concatenate(Seq(
      inputShape(0 :: -2),
      (inputShape(-2) * inputShape(-1)).expandDims(0))))
    if (input.rank != -1) {
      val lastDimSize = if (input.shape(-2) != -1 && input.shape(-1) != -1) input.shape(-2) * input.shape(-1) else -1
      val resultShape = input.shape(0 :: -2) ++ Shape(lastDimSize)
      result.setShape(resultShape)
    }
    result
  }

  /** Reshapes `input` so that the first two dimensions become one.
    *
    * @param  input Tensor with shape `[a, b, ...]`.
    * @return Tensor with shape `[a * b, ...]`.
    */
  def combineFirstTwoDimensions(input: Output): Output = {
    val result = tf.reshape(input, tf.concatenate(Seq(tf.constant(-1, shape = Shape(1)), tf.shape(input)(2 ::))))
    if (input.rank != -1) {
      val firstDimSize = if (input.shape(0) != -1 && input.shape(1) != -1) input.shape(0) * input.shape(1) else -1
      val resultShape = Shape(firstDimSize) ++ input.shape(2 ::)
      result.setShape(resultShape)
    }
    result
  }

  /** Shifts the second dimension of `input` right by one.
    *
    * @param  input        Input tensor.
    * @param  paddingValue Optional padding value to use.
    * @return Resulting tensor with the same shape as `input`.
    */
  def shift2DRight(input: Output, paddingValue: Option[Output] = None): Output = {
    // TODO: Make more generic.
    input.rank match {
      case 2 =>
        paddingValue.map(pv => {
          tf.concatenate(Seq(pv, input), axis = 1)(::, 0 :: -1)
        }).getOrElse {
          tf.pad(input, tf.stack(Seq(
            tf.stack(Seq(0, 0)),
            tf.stack(Seq(1, 0)))))(::, 0 :: -1)
        }
      case 3 =>
        paddingValue.map(pv => {
          tf.concatenate(Seq(pv, input), axis = 1)(::, 0 :: -1, ::)
        }).getOrElse {
          tf.pad(input, tf.stack(Seq(
            tf.stack(Seq(0, 0)),
            tf.stack(Seq(1, 0)),
            tf.stack(Seq(0, 0)))))(::, 0 :: -1, ::)
        }
      case 4 =>
        paddingValue.map(pv => {
          tf.concatenate(Seq(pv, input), axis = 1)(::, 0 :: -1, ::, ::)
        }).getOrElse {
          tf.pad(input, tf.stack(Seq(
            tf.stack(Seq(0, 0)),
            tf.stack(Seq(1, 0)),
            tf.stack(Seq(0, 0)),
            tf.stack(Seq(0, 0)))))(::, 0 :: -1, ::, ::)
        }
    }
  }

  /** Pads tensors `x` and `y` along the provided axis, so that they have the same length.
    *
    * @param  x                      First tensor to pad.
    * @param  y                      Second tensor to pad.
    * @param  finalLengthDivisibleBy The final length along the padded axis will be divisible by this number.
    * @param  axis                   Axis along which to pad.
    * @return Tuple containing the padded `x` and `y` tensors.
    */
  def padToSameLength(x: Output, y: Output, finalLengthDivisibleBy: Int = 1, axis: Int = 1): (Output, Output) = {
    require(axis == 1 || axis == 2, "The axis can only be set to 1 or 2.")
    tf.createWithNameScope("PadToSameLength") {
      val xLength = tf.shape(x)(axis)
      val yLength = tf.shape(y)(axis)
      var maxLength = tf.maximum(xLength, yLength)
      if (finalLengthDivisibleBy > 1) {
        // Find the nearest larger-or-equal integer divisible by the provided number.
        maxLength = tf.add(maxLength, finalLengthDivisibleBy - 1)
        maxLength = tf.truncateDivide(maxLength, finalLengthDivisibleBy)
        maxLength = tf.multiply(maxLength, finalLengthDivisibleBy)
      }

      def padding(lengthDiff: Output, arg: Output): Output = {
        val paddings = {
          if (axis == 1) {
            Seq(
              tf.stack(Seq(tf.stack(Seq(0, 0)), tf.stack(Seq(0, lengthDiff)))),
              tf.zeros(INT32, tf.stack(Seq(tf.rank(arg) - 2, 2))))
          } else {
            Seq(
              tf.stack(Seq(tf.stack(Seq(0, 0)), tf.stack(Seq(0, 0)), tf.stack(Seq(0, lengthDiff)))),
              tf.zeros(INT32, tf.stack(Seq(tf.rank(arg) - 3, 2))))
          }
        }
        tf.concatenate(paddings, axis = 0)
      }

      val xPadded = tf.pad(x, padding(maxLength - xLength, x))
      val yPadded = tf.pad(y, padding(maxLength - yLength, y))

      // Static shapes are the same except for axis 1.
      xPadded.setShape(Shape.fromSeq(x.shape.asArray.updated(axis, -1)))
      yPadded.setShape(Shape.fromSeq(y.shape.asArray.updated(axis, -1)))

      (xPadded, yPadded)
    }
  }

  /** Pads the provided labels along the length axis, so that it matches the provided logits length.
    *
    * @param  logits Logits tensor.
    * @param  labels Labels tensor.
    * @return Tuple containing the padded `logits` and `labels` tensors.
    */
  def padToSameLengthWithZeros(logits: Output, labels: Output): (Output, Output) = {
    tf.createWithNameScope("PadWithZeros") {
      var processed = padToSameLength(logits, labels)
      var processedLogits = processed._1
      var processedLabels = processed._2
      if (processedLabels.rank == 3) {
        // 2-D labels case.
        processed = padToSameLength(processedLogits, processedLabels, axis = 2)
        processedLogits = processed._1
        processedLabels = processed._2
      }
      (processedLogits, processedLabels)
    }
  }

  //region Weight Functions

  /** Assigns weight `1.0f` to all labels.
    *
    * @param  labels Target labels.
    * @return  `FLOAT32` tensor containing weights for the provided labels.
    */
  def weightsAll(labels: Output): Output = {
    tf.onesLike(labels, FLOAT32)
  }

  /** Assigns weight `1.0f` to all labels except for those equal to `0` (i.e., padding).
    *
    * @param  labels Target labels.
    * @return  `FLOAT32` tensor containing weights for the provided labels.
    */
  def weightsNonZero(labels: Output): Output = {
    tf.notEqual(labels, 0).cast(FLOAT32)
  }

  /** Assign weight `1.0f` to only the "targets" portion of the labels. Weight `1.0f` is assigned to all nonzero labels
    * past the first zero.
    *
    * @param  labels Target labels.
    * @return  `FLOAT32` tensor containing weights for the provided labels.
    */
  def weightsPrependInputsToTargets(labels: Output): Output = {
    val pastFirstZero = tf.cumsum(tf.equal(labels, 0).cast(FLOAT32), axis = 1)
    tf.notEqual(pastFirstZero * labels.cast(FLOAT32), 0).cast(FLOAT32)
  }

  /** Assigns weight `1.0f` to only the "target" part of the concatenated labels.
    *
    * The labels look like:
    *   source English I love you . EOS target French Je t'aime . EOS
    *   source English the cat EOS target French le chat EOS
    *   source English ...
    *
    * We want to assign weight `1.0f` to all words in the target text (including the EOS symbol), but not to the source
    * text or the boilerplate. In the above example, the target words that get positive weight are:
    *   Je t'aime . EOS le chat EOS
    *
    * @param  labels Target labels.
    * @return  `FLOAT32` tensor containing weights for the provided labels.
    */
  def weightsConcatenated(labels: Output): Output = {
    val eosMask = tf.equal(labels, 1).cast(INT32) // TODO: Standardize EOS symbol.
    val sentenceNum = tf.cumsum(eosMask, axis = 1, exclusive = true)
    val sentenceNumPlusOne = sentenceNum + 1
    val inTarget = tf.equal(tf.mod(sentenceNum, 2), 1)
    // The first two tokens of each sentence are boilerplate.
    val shifted = tf.pad(sentenceNumPlusOne, tf.stack(
      Seq(tf.stack(Seq(0, 0)), tf.stack(Seq(2, 0)), tf.stack(Seq(0, 0)), tf.stack(Seq(0, 0)))))(::, 0 :: -2, ::, ::)
    val nonBoilerplate = tf.equal(sentenceNumPlusOne, shifted)
    tf.logicalAnd(nonBoilerplate, inTarget).cast(FLOAT32)
  }

  //endregion Weight Functions

  //region Loss Functions

  /** Computes the cross-entropy loss between `logits` and `labels` using label smoothing to limit over-confidence.
    *
    * @param  logits         Tensor with shape `[batchSize, ..., vocabSize]`.
    * @param  labels         Tensor with shape `[batchSize, ...]`.
    * @param  labelSmoothing Value used to determine on and off values for label smoothing. If `gaussian` is `true`,
    *                        `confidence = 1.0f - labelSmoothing` is the variance of the gaussian distribution.
    * @param  gaussian       If `true`, a Gaussian distribution will be used for label smoothing.
    * @return Cross entropy scores.
    */
  def smoothingCrossEntropy(
      logits: Output,
      labels: Output,
      labelSmoothing: Float = 0.0f,
      gaussian: Boolean = false
  ): Output = {
    tf.createWithNameScope("SmoothingCrossEntropy") {
      val vocabSize = tf.shape(logits)(-1)
      val vocabSizeMinusOne = (vocabSize - 1).cast(FLOAT32)
      // Low confidence is given to all non-true labels, uniformly.
      val confidence = 1.0f - labelSmoothing
      val lowConfidence = (1.0f - confidence) / vocabSizeMinusOne
      // The normalizing constant is the best cross-entropy value with soft targets.
      // We subtract it just for readability purposes, as it makes no difference to the learning process.
      val normalizingConstant = -(confidence * math.log(confidence).toFloat +
          vocabSizeMinusOne * lowConfidence * tf.log(lowConfidence + 1e-20f))
      val softTargets = {
        if (gaussian) {
          ???
          // TODO: Gaussian:
          //
          //      labels = tf.cast(labels, tf.float32)
          //      normal_dist = tf.distributions.Normal(loc=labels, scale=confidence)
          //      # Locations to evaluate the probability distributions.
          //      soft_targets = normal_dist.prob(
          //          tf.cast(tf.range(vocab_size), tf.float32)[:, None, None, None, None])
          //      # Reordering soft_targets from [vocab_size, batch_size, ?, ?, ?] to match
          //      # logits: [batch_size, ?, ?, ?, vocab_size]
          //      soft_targets = tf.transpose(soft_targets, perm=[1, 2, 3, 4, 0])
        } else {
          tf.oneHot(tf.cast(labels, INT64), vocabSize, confidence, lowConfidence)
        }
      }
      tf.softmaxCrossEntropy(logits, softTargets) - normalizingConstant
    }
  }

  /** Computes the cross-entropy loss between `logits` and `labels` using label smoothing to limit over-confidence,
    * while assuming that `0` labels correspond to padding.
    *
    * @param  logits         Tensor with shape `[batchSize, ..., vocabSize]`.
    * @param  labels         Tensor with shape `[batchSize, ...]`.
    * @param  labelLengths   Tensor with shape `[batchSize]`, containing the lengths of the target sequences.
    * @param  labelSmoothing Value used to determine on and off values for label smoothing. If `gaussian` is `true`,
    *                        `confidence = 1.0f - labelSmoothing` is the variance of the gaussian distribution.
    * @param  sum            If `true`, the individual sample cross-entropies and weights are summed before returned.
    * @param  gaussian       If `true`, a Gaussian distribution will be used for label smoothing.
    * @return Tuple containing the cross-entropy tensor and the weights tensor.
    */
  def paddedCrossEntropy(
      logits: Output,
      labels: Output,
      labelLengths: Output,
      labelSmoothing: Float = 0.0f,
      sum: Boolean = true,
      gaussian: Boolean = false,
      timeMajor: Boolean = false
  ): (Output, Output) = {
    // TODO: Factored padded cross-entropy.
    tf.createWithNameScope("PaddedCrossEntropy") {
      val maxLength = tf.shape(labels)(1)
      // val (processedLogits, processedLabels) = padToSameLengthWithZeros(logits, labels)
      val transposedLabels = if (timeMajor) labels.transpose() else labels
      val crossEntropy = smoothingCrossEntropy(logits, transposedLabels, labelSmoothing, gaussian)
      val weights = tf.sequenceMask(labelLengths, maxLength, logits.dataType)
      val transposedWeights = if (timeMajor) weights.transpose() else weights
      if (!sum)
        (crossEntropy * transposedWeights, transposedWeights)
      else
        (tf.sum(crossEntropy * transposedWeights), tf.sum(transposedWeights))
    }
  }

  //endregion Loss Functions

  /** Computes a dropout layer.
    *
    * With probability `keepProbability`, the op outputs the input element scaled up by `1 / keepProbability`,
    * otherwise the created op outputs `0`. The scaling is such that the expected sum remains unchanged.
    *
    * By default, each element is kept or dropped independently. If `broadcastAxes` is specified, it defines which axes
    * (i.e., dimensions) of the `input` tensor will be dropped out together (i.e., the dropout decision is broadcast
    * along these axes).
    *
    * @param  input           Input tensor.
    * @param  keepProbability Probability (i.e., number in the interval `(0, 1]`) that each element is kept.
    * @param  scaleOutput     If `true`, the outputs will be divided by the keep probability.
    * @param  broadcastAxes   Specifies along which axes the dropout is broadcast.
    * @return Created op output that has the same shape as `input`.
    */
  def dropoutWithBroadcastAxes(
      input: Output,
      keepProbability: Float,
      scaleOutput: Boolean = true,
      broadcastAxes: Set[Int] = Set.empty
  ): Output = {
    val one = tf.constant(1, dataType = INT32)
    val noiseShape = {
      if (broadcastAxes.isEmpty) {
        tf.shape(input)
      } else {
        val inputShape = tf.shape(input)
        tf.stack((0 until input.rank).map(i => {
          if (broadcastAxes.contains(i))
            one
          else
            inputShape(i)
        }))
      }
    }
    tf.dropout(input, keepProbability, scaleOutput, noiseShape)
  }

  /** Applies a fully connected layer to `input`, followed by a ReLU activation, optionally a dropout, and finally,
    * another fully connected layer.
    *
    * @param  input                    Input tensor.
    * @param  filterSize               First layer projection size.
    * @param  outputSize               Output layer projection size.
    * @param  reluDropoutRate          Dropout rate for the output of the ReLU activation.
    * @param  reluDropoutBroadcastAxes Specifies along which axes of the attention weights the dropout is broadcast.
    * @param  name                     Name for this alayer that also specifies a variable scope.
    * @param  mode                     Current learning mode (e.g., training or evaluation).
    * @param  parameterManager        Parameter manager to use, if parameters are required.
    * @return Output of the last fully connected layer.
    */
  def denseReLUDense(
      input: Output,
      filterSize: Int,
      outputSize: Int,
      reluDropoutRate: Float = 0.0f,
      reluDropoutBroadcastAxes: Set[Int] = Set.empty,
      name: String = "DenseReLUDense"
  )(mode: Mode, parameterManager: ParameterManager)(implicit
      stage: Stage
  ): Output = tf.variableScope(name) {
    val weights1 = parameterManager.get("Dense1/Weights", input.dataType, Shape(input.shape(-1), filterSize))
    val bias1 = parameterManager.get("Dense1/Bias", input.dataType, Shape(filterSize))
    var hidden = tf.relu(tf.linear(input, weights1, bias1, "Dense1"), name = "Dense1/ReLU")
    if (mode.isTraining)
      hidden = dropoutWithBroadcastAxes(hidden, 1.0f - reluDropoutRate, scaleOutput = true, reluDropoutBroadcastAxes)
    val weights2 = parameterManager.get("Dense2/Weights", input.dataType, Shape(filterSize, outputSize))
    val bias2 = parameterManager.get("Dense2/Bias", input.dataType, Shape(outputSize))
    tf.linear(hidden, weights2, bias2, "Dense2")
  }
}
