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

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}

/**
  * @author Emmanouil Antonios Platanios
  */
trait PositionalEmbeddings {
  // TODO: !!! Add name to positional embeddings.

  def get(
      length: Output[Int],
      depth: Output[Int]
  ): Output[Float]

  def addTo[T: TF : IsNotQuantized](
      input: Output[T],
      positions: Option[Output[Int]] = None
  ): Output[T]
}

object PositionalEmbeddings {
  /** Creates a bunch of sinusoids of different frequencies and phases, to be used as positional embeddings.
    *
    * This allows attention models to learn and use absolute and relative positions. Positional embeddings should be
    * added to some precursors of both the query and the memory inputs to attention.
    *
    * The use of relative position is possible because `sin(x + y)` and `cos(x + y)` can be expressed in terms of `y`,
    * `sin(x)` and `cos(x)`.
    *
    * In particular, we use a geometric sequence of scales starting with `minScale` and ending with `maxScale`. The
    * number of different scales is equal to `depth / 2`. For each scale, we generate the two sinusoidal signals
    * `sin(position / scale)` and `cos(position / scale)`. All of these sinusoids are concatenated along the depth
    * dimension.
    *
    * @param  length   Length of the input sequences.
    * @param  depth    Depth of the input sequences (i.e., number of channels).
    * @param  minScale Minimum scale.
    * @param  maxScale Maximum scale.
    * @return Positional embeddings tensor with shape `[1, length, depth]`.
    */
  def positionalEmbeddings1D(
      length: Output[Int],
      depth: Output[Int],
      minScale: Float = 1.0f,
      maxScale: Float = 1.0e4f
  ): Output[Float] = {
    val zero = tf.constant[Int](0)
    val one = tf.constant[Int](1)
    val two = tf.constant[Int](2)
    val positions = tf.range(zero, length).toFloat
    val numScales = tf.truncateDivide(depth, two)
    val logScaleIncrement = math.log(maxScale / minScale).toFloat / (numScales - one).toFloat
    val invScales = minScale * tf.exp(-tf.range(zero, numScales).toFloat * logScaleIncrement)
    val scaledPositions = tf.expandDims(positions, axis = one) * tf.expandDims(invScales, axis = zero)
    var embedding = tf.concatenate(Seq(tf.sin(scaledPositions), tf.cos(scaledPositions)), axis = one)
    val padding = tf.stack(Seq(tf.stack(Seq(zero, zero)), tf.stack(Seq(zero, tf.mod(depth, two)))))
    embedding = tf.pad(embedding, padding)
    embedding = embedding.reshape(tf.stack(Seq(one, length, depth)))
    embedding
  }

  /** Adds a bunch of sinusoids of different frequencies and phases to `input`.
    *
    * Each channel (i.e., 3rd dimension) of the input tensor is incremented by a sinusoid of a different frequency and
    * phase.
    *
    * This allows attention models to learn and use absolute and relative positions. Positional embeddings should be
    * added to some precursors of both the query and the memory inputs to attention.
    *
    * The use of relative position is possible because `sin(x + y)` and `cos(x + y)` can be expressed in terms of `y`,
    * `sin(x)` and `cos(x)`.
    *
    * In particular, we use a geometric sequence of scales starting with `minScale` and ending with `maxScale`. The
    * number of different scales is equal to `depth / 2`. For each scale, we generate the two sinusoidal signals
    * `sin(position / scale)` and `cos(position / scale)`. All of these sinusoids are concatenated along the depth
    * dimension.
    *
    * @param  input    Input tensor with shape `[batchSize, length, depth]`.
    * @param  minScale Minimum scale.
    * @param  maxScale Maximum scale.
    * @return Input tensor with the positional embeddings added to it.
    */
  def addPositionalEmbeddings1D[T: TF : IsNotQuantized](
      input: Output[T],
      minScale: Float = 1.0f,
      maxScale: Float = 1.0e4f
  ): Output[T] = {
    val inputShape = tf.shape(input).toInt
    val length = inputShape(1)
    val depth = inputShape(2)
    val embedding = positionalEmbeddings1D(length, depth, minScale, maxScale)
    input + embedding.castTo[T]
  }

  /** Adds a bunch of sinusoids of different frequencies and phases to `input`, using the provided positions.
    *
    * @param  input     Input tensor with shape `[batchSize, length, depth]`.
    * @param  positions Positions tensor with shape `[batchSize, length]`.
    * @param  minScale  Minimum scale.
    * @param  maxScale  Maximum scale.
    * @return Input tensor with the positional embeddings added to it.
    */
  def addPositionalEmbeddings1DGivenPositions[T: TF : IsNotQuantized](
      input: Output[T],
      positions: Output[Int],
      minScale: Float = 1.0f,
      maxScale: Float = 1.0e4f
  ): Output[T] = {
    val inputShape = tf.shape(input).toInt
    val depth = inputShape(2).toInt
    val zero = tf.constant[Int](0)
    val one = tf.constant[Int](1)
    val two = tf.constant[Int](2)
    val numScales = tf.truncateDivide(depth, two).toInt
    val logScaleIncrement = math.log(maxScale / minScale).toFloat / (numScales - one).toFloat
    val invScales = minScale * tf.exp(-tf.range(zero, numScales).toFloat * logScaleIncrement)
    val scaledPositions = tf.expandDims(positions.toFloat, axis = two) *
        tf.expandDims(tf.expandDims(invScales, axis = zero), axis = zero)
    val embedding = tf.concatenate(Seq(tf.sin(scaledPositions), tf.cos(scaledPositions)), axis = two)
    val padding = tf.stack(Seq(
      tf.stack(Seq(zero, zero)),
      tf.stack(Seq(zero, zero)),
      tf.stack(Seq(zero, tf.mod(depth, two).toInt))))
    input + tf.pad(embedding, padding).castTo[T]
  }

  /** Adds a bunch of sinusoids of different frequencies and phases to `input`.
    *
    * Each channel (i.e., 3rd dimension) of the input tensor is incremented by a sinusoid of a different frequency and
    * phase.
    *
    * This allows attention models to learn and use absolute and relative positions. Positional embeddings should be
    * added to some precursors of both the query and the memory inputs to attention.
    *
    * The use of relative position is possible because `sin(x + y)` and `cos(x + y)` can be expressed in terms of `y`,
    * `sin(x)` and `cos(x)`.
    *
    * `input` is a tensor with `n` "positional" dimensions (e.g., one for a sequence or two for an image).
    *
    * In particular, we use a geometric sequence of scales starting with `minScale` and ending with `maxScale`. The
    * number of different scales is equal to `depth / (n * 2)`. For each scale, we generate the two sinusoidal signals
    * `sin(position / scale)` and `cos(position / scale)`. All of these sinusoids are concatenated along the depth
    * dimension.
    *
    * @param  input    Input tensor with shape `[batchSize, d1, ..., dn, depth]`.
    * @param  minScale Minimum scale.
    * @param  maxScale Maximum scale.
    * @return Input tensor with the positional embeddings added to it.
    */
  def addPositionalEmbeddingsND[T: TF : IsNotQuantized](
      input: Output[T],
      minScale: Float = 1.0f,
      maxScale: Float = 1.0e4f
  ): Output[T] = {
    val rank = input.rank - 2
    val inputShape = tf.shape(input).toInt
    val depth = inputShape(-1)
    val zero = tf.constant[Int](0)
    val one = tf.constant[Int](1)
    val two = tf.constant[Int](2)
    val minusTwo = tf.constant[Int](-2)
    val numScales = tf.truncateDivide(depth, two * rank).toInt
    val logScaleIncrement = math.log(maxScale / minScale).toFloat / (numScales - one).toFloat
    val invScales = minScale * tf.exp(-tf.range(zero, numScales).toFloat * logScaleIncrement)
    var result = input
    (0 until rank).foreach(axis => {
      val length = inputShape(axis + 1)
      val positions = tf.range(zero, length).toFloat
      val scaledPositions = tf.expandDims(positions, axis = one) * tf.expandDims(invScales, axis = zero)
      var embedding = tf.concatenate(Seq(tf.sin(scaledPositions), tf.cos(scaledPositions)), axis = one)
      val prePadding = 2 * rank * numScales
      val postPadding = depth - (axis + 1) * 2 * numScales
      embedding = tf.pad(embedding, tf.stack(Seq(
        tf.stack(Seq(zero, zero)),
        tf.stack(Seq(prePadding, postPadding))
      )))
      (0 until axis + 1).foreach(_ => embedding = tf.expandDims(embedding, axis = zero))
      (0 until rank - 1 - axis).foreach(_ => embedding = tf.expandDims(embedding, axis = minusTwo))
      result += embedding.castTo[T]
    })
    result
  }
}

class FixedSinusoidPositionalEmbeddings protected (
    val minScale: Float = 1.0f,
    val maxScale: Float = 1.0e4f
) extends PositionalEmbeddings {
  override def get(
      length: Output[Int],
      depth: Output[Int]
  ): Output[Float] = {
    PositionalEmbeddings.positionalEmbeddings1D(length, depth, minScale, maxScale)
  }

  override def addTo[T: TF : IsNotQuantized](
      input: Output[T],
      positions: Option[Output[Int]] = None
  ): Output[T] = {
    positions match {
      case Some(p) => PositionalEmbeddings.addPositionalEmbeddings1DGivenPositions(input, p, minScale, maxScale)
      case None => PositionalEmbeddings.addPositionalEmbeddings1D(input, minScale, maxScale)
    }
  }
}

object FixedSinusoidPositionalEmbeddings {
  def apply(minScale: Float = 1.0f, maxScale: Float = 1.0e4f): FixedSinusoidPositionalEmbeddings = {
    new FixedSinusoidPositionalEmbeddings(minScale, maxScale)
  }
}
