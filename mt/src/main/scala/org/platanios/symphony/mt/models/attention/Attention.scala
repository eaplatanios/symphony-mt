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

package org.platanios.symphony.mt.models.attention

import org.platanios.symphony.mt.models.ParametersManager
import org.platanios.symphony.mt.models.helpers.Common
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode

import scala.language.postfixOps

/**
  * @author Emmanouil Antonios Platanios
  */
trait Attention {
  /** Computes the attention for the provided queries, keys, and values.
    *
    * @param  q                 Queries tensor with shape `[batchSize, ..., length, depth]`.
    * @param  k                 Keys tensor with shape `[batchSize, ..., length, depth]`.
    * @param  v                 Values tensor with shape `[batchSize, ..., length, depth]`.
    * @param  bias              Optional attention bias.
    * @param  mode              Current learning mode (e.g., training or evaluation).
    * @param  parametersManager Parameter manager to use, if parameters are required.
    * @return Attention tensor with shape `[batchSize, ..., length, depth]`.
    */
  def apply(
      q: Output,
      k: Output,
      v: Output,
      bias: Option[Output]
  )(mode: Mode, parametersManager: ParametersManager[_, _]): Output

  // TODO: Add support for saving weights.
  // TODO: Add support for image summaries for the weights.
}

object Attention {
  //region Attention Bias

  /** Creates a bias tensor to be added to the attention logits.
    *
    * A position may attend to positions of at most some distance from it (forwards and backwards).
    *
    * @param  length      Integer scalar representing the sequence length.
    * @param  maxBackward Maximum backwards distance to attend. Negative values indicate an unlimited distance.
    * @param  maxForward  Maximum forwards distance to attend. Negative values indicate an unlimited distance.
    * @return `FLOAT32` tensor with shape `[1, 1, length, length]`.
    */
  def attentionBiasLocal(length: Output, maxBackward: Int, maxForward: Int): Output = {
    var band = tf.matrixBandPart(
      tf.ones(FLOAT32, tf.stack(Seq(length, length))),
      tf.constant(maxBackward, INT64),
      tf.constant(maxForward, INT64))
    band = tf.reshape(band, tf.stack(Seq(1, 1, length, length)))
    tf.cast(-1e9f * (1.0f - band), FLOAT32)
  }

  /** Creates a bias tensor to be added to the attention logits.
    *
    * This bias allows a query to attend to all positions up to and including its own.
    *
    * @param  length Integer scalar representing the sequence length.
    * @return `FLOAT32` tensor with shape `[1, 1, length, length]`.
    */
  def attentionBiasLowerTriangular(length: Output): Output = {
    attentionBiasLocal(length, -1, 0)
  }

  /** Creates a bias tensor to be added to the attention logits.
    *
    * This bias allows positions with the same segment IDs to see each other.
    *
    * @param  querySegmentIDs  Tensor with shape `[batchSize, queryLength]`.
    * @param  memorySegmentIDs Tensor with shape `[batchSize, memoryLength]`.
    * @return `FLOAT32` tensor with shape `[batchSize, 1, queryLength, memoryLength]`.
    */
  def attentionBiasSameSegment(querySegmentIDs: Output, memorySegmentIDs: Output): Output = {
    tf.expandDims(tf.notEqual(
      tf.expandDims(querySegmentIDs, axis = 2),
      tf.expandDims(memorySegmentIDs, axis = 1)).cast(FLOAT32) * -1e9f, axis = 1)
  }

  /** Creates a bias tensor to be added to the attention logits.
    *
    * @param  padding `FLOAT32` tensor with shape `[batchSize, length]`.
    * @param  epsilon Bias value to the be added.
    * @return Bias tensor with shape `[batchSize, 1, 1, length]`.
    */
  def attentionBiasIgnorePadding(padding: Output, epsilon: Float = -1e9f): Output = {
    tf.expandDims(tf.expandDims(padding * epsilon, axis = 1), axis = 1)
  }

  /** Creates a bias tensor to be added to the attention logits.
    *
    * @param  attentionBias Tensor with shape `[batchSize, 1, 1, length]`.
    * @return Bias tensor with shape `[batchSize, length]`, with `1.0f` in padding positions and `0.0f` in non-padding
    *         positions.
    */
  def attentionBiasToPadding(attentionBias: Output): Output = {
    // `attentionBias` is a large negative number in padding positions and zero elsewhere.
    tf.squeeze(tf.less(attentionBias, -1).cast(FLOAT32), axes = Seq(1, 2))
  }

  /** Creates a bias tensor for self-attention, that encourages attention to close positions.
    *
    * @param  length Integer scalar representing the sequence length.
    * @return Bias tensor with shape `[1, 1, length, length]`.
    */
  def attentionBiasProximal(length: Output): Output = {
    val r = tf.range(0, length).cast(FLOAT32)
    val diff = tf.expandDims(r, 0) - tf.expandDims(r, 1)
    tf.expandDims(tf.expandDims(-tf.log(1 + tf.abs(diff)), axis = 0), axis = 0)
  }

  /** Creates a bias tensor to be added to the attention logits.
    *
    * This bias prevents batches from attending to each other.
    *
    * @param  conditionFn       Function defining which type of mask to build.
    * @param  batchCoordinatesQ `INT32` tensor with shape `[lengthQ, 1]`, containing the coordinates of the batches.
    * @param  batchCoordinatesK `INT32` tensor with shape `[lengthQ, 1]`, containing the coordinates of the batches.
    * @return `FLOAT32` tensor with shape `[lengthQ, lengthK]`, containing either `0` or `-infinity` (i.e., `-1e9f`).
    */
  def attentionBiasBatch(
      conditionFn: (Output) => Output,
      batchCoordinatesQ: Output,
      batchCoordinatesK: Output
  ): Output = {
    val bcQ = tf.expandDims(tf.squeeze(batchCoordinatesQ, axes = Seq(1)).cast(FLOAT32), axis = 1)
    val bcK = tf.expandDims(tf.squeeze(batchCoordinatesK, axes = Seq(1)).cast(FLOAT32), axis = 0)
    // Broadcast to create a `[lengthQ, lengthK]` mask.
    conditionFn(bcK - bcQ) * -1e9f
  }

  /** Creates a bias tensor to be added to the attention logits.
    *
    * This bias prevents individual sequences of the same batch from attending to each other.
    *
    * @param  batchCoordinatesQ `INT32` tensor with shape `[lengthQ, 1]`, containing the coordinates of the batches.
    * @param  batchCoordinatesK `INT32` tensor with shape `[lengthQ, 1]`, containing the coordinates of the batches.
    * @return `FLOAT32` tensor with shape `[lengthQ, lengthK]`, containing either `0` or `-infinity` (i.e., `-1e9f`).
    */
  def attentionBiasCoordinates(batchCoordinatesQ: Output, batchCoordinatesK: Output): Output = {
    attentionBiasBatch(bias => tf.minimum(1.0f, tf.abs(bias)), batchCoordinatesQ, batchCoordinatesK)
  }

  /** Creates a bias tensor to be added to the attention logits.
    *
    * This bias prevents individual sequences of the same batch from attending to future values.
    *
    * @param  batchCoordinatesQ `INT32` tensor with shape `[lengthQ, 1]`, containing the coordinates of the batches.
    * @param  batchCoordinatesK `INT32` tensor with shape `[lengthQ, 1]`, containing the coordinates of the batches.
    * @return `FLOAT32` tensor with shape `[lengthQ, lengthK]`, containing either `0` or `-infinity` (i.e., `-1e9f`).
    */
  def attentionBiasFuture(batchCoordinatesQ: Output, batchCoordinatesK: Output): Output = {
    attentionBiasBatch(bias => tf.maximum(0.0f, tf.minimum(1.0f, tf.abs(bias))), batchCoordinatesQ, batchCoordinatesK)
  }

  //endregion Attention Bias

  // TODO: encoderDecoderAttentionLoss

  //region Multi-Head Attention

  /** Splits the third dimension of `input` into multiple heads (becomes the second dimension).
    *
    * @param  input    Tensor with shape `[batchSize, length, depth]`.
    * @param  numHeads Number of heads to split in.
    * @return Tensor with shape `[batchSize, numHeads, length, depth / numHeads]`.
    */
  def splitHeads(input: Output, numHeads: Int): Output = {
    tf.transpose(Common.splitLastDimension(input, numHeads), Seq(0, 2, 1, 3))
  }

  /** Splits the fourth dimension of `input` into multiple heads (becomes the second dimension).
    *
    * @param  input    Tensor with shape `[batchSize, height, width, depth]`.
    * @param  numHeads Number of heads to split in.
    * @return Tensor with shape `[batchSize, numHeads, height, width, depth / numHeads]`.
    */
  def splitHeads2D(input: Output, numHeads: Int): Output = {
    tf.transpose(Common.splitLastDimension(input, numHeads), Seq(0, 3, 1, 2, 4))
  }

  /** Inverse of `splitHeads`.
    *
    * @param  input Tensor with shape `[batchSize, numHeads, length, depth / numHeads]`.
    * @return Tensor with shape `[batchSize, length, depth]`.
    */
  def combineHeads(input: Output): Output = {
    Common.combineLastTwoDimensions(tf.transpose(input, Seq(0, 2, 1, 3)))
  }

  /** Inverse of `splitHeads2D`.
    *
    * @param  input Tensor with shape `[batchSize, numHeads, height, width, depth / numHeads]`.
    * @return Tensor with shape `[batchSize, height, width, depth]`.
    */
  def combineHeads2D(input: Output): Output = {
    Common.combineLastTwoDimensions(tf.transpose(input, Seq(0, 2, 3, 1, 4)))
  }

  /** Computes the query, key, and value tensors, for attention models.
    *
    * @param  queryAntecedent   Tensor with shape `[batch, queryLength, depth]`.
    * @param  memoryAntecedent  Tensor with shape `[batch, memoryLength, depth]`.
    * @param  totalKeysDepth    Depth for the projected keys (and queries).
    * @param  totalValuesDepth  Depth for the projected values.
    * @param  qNumFilters       Integer specifying how wide we want the queries to be.
    * @param  kvNumFilters      Integer specifying how wide we want the keys and values to be.
    * @param  qPaddingMode      Convolution padding mode for the case when `qNumFilters > 1`.
    * @param  kvPaddingMode     Convolution padding mode for the case when `kvNumFilters > 1`.
    * @param  mode              Current learning mode (e.g., training or evaluation).
    * @param  parametersManager Parameter manager to use, if parameters are required.
    * @return Tuple containing the queries, keys, and values tensors.
    */
  def computeQKV(
      queryAntecedent: Output,
      memoryAntecedent: Output,
      totalKeysDepth: Int,
      totalValuesDepth: Int,
      qNumFilters: Int = 1,
      kvNumFilters: Int = 1,
      qPaddingMode: tf.ConvPaddingMode = tf.ValidConvPadding,
      kvPaddingMode: tf.ConvPaddingMode = tf.ValidConvPadding
  )(mode: Mode, parametersManager: ParametersManager[_, _]): (Output, Output, Output) = {

    def compute(input: Output, depth: Int, numFilters: Int, paddingMode: tf.ConvPaddingMode, name: String): Output = {
      tf.createWithVariableScope(name) {
        if (numFilters == 1) {
          val weights = parametersManager.get("Weights", input.dataType, Shape(input.shape(-1), depth))
          tf.linear(input, weights)
        } else {
          ???
        }
      }
    }

    val q = compute(queryAntecedent, totalKeysDepth, qNumFilters, qPaddingMode, "Q")
    val k = compute(memoryAntecedent, totalKeysDepth, kvNumFilters, kvPaddingMode, "K")
    val v = compute(memoryAntecedent, totalValuesDepth, kvNumFilters, kvPaddingMode, "V")
    (q, k, v)
  }

  /** Applies multi-head attention using input and output projections.
    *
    * '''NOTE:''' For decoder self-attention, (i.e. when `memoryAntecedent == queryAntecedent`, the caching assumes that
    * the bias contains future masking. Caching works by saving all the previous key and value values so that you are
    * able to send just the last query location to this attention function. I.e., if the cache is provided it assumes
    * that the query has shape `[batchSize, 1, outputDepth]` rather than the full sequence.
    *
    * @param  queryAntecedent   Tensor with shape `[batch, queryLength, depth]`.
    * @param  memoryAntecedent  Tensor with shape `[batch, memoryLength, depth]`.
    * @param  bias              Attention bias tensor.
    * @param  totalKeysDepth    Total depth for the projected keys (concatenated over all heads).
    * @param  totalValuesDepth  Total depth for the projected values (concatenated over all heads).
    * @param  outputsDepth      Depth for the projected outputs.
    * @param  numHeads          Number of heads to use. Must divide `totalKeyDepth` and `totalValueDepth`.
    * @param  attention         Attention model to use as the main building block of this multi-head attention wrapper.
    * @param  qNumFilters       Integer specifying how wide we want the queries to be.
    * @param  kvNumFilters      Integer specifying how wide we want the keys and values to be.
    * @param  qPaddingMode      Convolution padding mode for the case when `qNumFilters > 1`.
    * @param  kvPaddingMode     Convolution padding mode for the case when `kvNumFilters > 1`.
    * @param  cache             Optional cache containing the result of previous attentions, used for fast decoding.
    *                           For the initial call, the values for these keys should be
    * @param  name              Name for the multi-head attention component that also specifies a variable scope.
    * @param  mode              Current learning mode (e.g., training or evaluation).
    * @param  parametersManager Parameter manager to use, if parameters are required.
    * @return Result of the attention transformation, with shape `[batchSize, queryLength, outputDepth]`, unless a cache
    *         is provided, in which case only the last memory position is calculated and the output shape is
    *         `[batchSize, 1, outputDepth]`.
    * @throws IllegalArgumentException If a cache is provided, but the attention model is not `DotProductAttention`, of
    *                                  if a cache is provided, but no `bias` is provided.
    */
  @throws[IllegalArgumentException]
  def multiHeadAttention(
      queryAntecedent: Output,
      memoryAntecedent: Output,
      bias: Output,
      totalKeysDepth: Int,
      totalValuesDepth: Int,
      outputsDepth: Int,
      numHeads: Int,
      attention: Attention,
      qNumFilters: Int = 1,
      kvNumFilters: Int = 1,
      qPaddingMode: tf.ConvPaddingMode = tf.ValidConvPadding,
      kvPaddingMode: tf.ConvPaddingMode = tf.ValidConvPadding,
      cache: Option[Cache] = None,
      name: String = "MultiHeadAttention"
  )(mode: Mode, parametersManager: ParametersManager[_, _]): Output = {
    require(totalKeysDepth % numHeads == 0, "`totalKeyDepth` must be divisible by `numHeads`.")
    require(totalValuesDepth % numHeads == 0, "`totalValueDepth` must be divisible by `numHeads`.")
    tf.createWithVariableScope(name) {
      var (q, k, v) = computeQKV(
        queryAntecedent, memoryAntecedent, totalKeysDepth, totalValuesDepth, qNumFilters, kvNumFilters,
        qPaddingMode, kvPaddingMode)(mode, parametersManager)
      cache match {
        case Some(c) =>
          require(
            attention.isInstanceOf[DotProductAttention],
            "Caching is not guaranteed to work with attention types other than 'DotProductAttention'.")
          require(bias != null, "Bias is required for caching.")
          k = tf.concatenate(Seq(c.k, k), axis = 1)
          v = tf.concatenate(Seq(c.v, v), axis = 1)
          c.k = k
          c.v = v
        case None => ()
      }
      q = splitHeads(q, numHeads)
      k = splitHeads(k, numHeads)
      v = splitHeads(v, numHeads)
      q = q * tf.pow(tf.truncateDivide(totalKeysDepth, numHeads), -0.5f)
      var result = attention(q, k, v, Some(bias))(mode, parametersManager)
      result = combineHeads(result)
      val w = parametersManager.get("OutputTransformWeights", result.dataType, Shape(result.shape(-1), outputsDepth))
      tf.linear(result, w)
    }
  }

  class Cache protected(var k: Output, var v: Output)

  object Cache {
    def apply(k: Output, v: Output): Cache = {
      new Cache(k, v)
    }
  }

  //endregion Multi-Head Attention
}
