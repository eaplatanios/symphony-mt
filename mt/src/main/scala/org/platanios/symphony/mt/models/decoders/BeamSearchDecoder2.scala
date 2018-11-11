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

// package org.platanios.symphony.mt.models.helpers.decoders

// import org.platanios.tensorflow.api._
// import org.platanios.tensorflow.api.core.Indexer.NewAxis
// import org.platanios.tensorflow.api.core.exception.{InvalidArgumentException, InvalidShapeException}
// import org.platanios.tensorflow.api.implicits.Implicits._
// import org.platanios.tensorflow.api.implicits.helpers.OutputStructure

// import scala.collection.mutable.ArrayBuffer
// import scala.language.postfixOps

// /**
//   * @author Emmanouil Antonios Platanios
//   */
// object BeamSearchDecoder2 {
//   /** Merges the first two dimensions of `value` into one.
//     *
//     * Typically, the first dimension corresponds to the batch size and the second to the beam width. So, for example,
//     * if `value` has shape `[batchSize, beamWidth, ...]`, the resulting tensor will have shape
//     * `[batchSize * beamWidth, ...]`.
//     *
//     * @param  value Tensor to reshape.
//     * @return `value` reshaped such that its first two dimensions are merged into one.
//     */
//   private[BeamSearchDecoder2] def mergeBeams[T: TF](
//       value: Output[T]
//   ): Output[T] = {
//     val valueShape = Basic.shape(value)
//     val batchSize = valueShape(0)
//     val beamWidth = valueShape(1)
//     val mergedSize = batchSize * beamWidth
//     val mergedShape = Basic.concatenate(Seq(mergedSize(NewAxis), valueShape(2 ::)), axis = 0)
//     Basic.reshape(value, mergedShape)
//   }

//   /** Splits the first dimension of `value` into two, reversing the merge operation of `mergeBeams`.
//     *
//     * For example, if `value` has shape `[batchSize * beamWidth, ...]`, the resulting tensor will have shape
//     * `[batchSize, beamWidth, ...]`.
//     *
//     * @param  value     Tensor to reshape.
//     * @param  batchSize Original batch size.
//     * @param  beamWidth Original beam width.
//     * @return `value` reshaped such that its first dimension are split into two.
//     */
//   private[BeamSearchDecoder2] def splitBeams[T: TF](
//       value: Output[T],
//       batchSize: Output[Int],
//       beamWidth: Output[Int]
//   ): Output[T] = {
//     val valueShape = Basic.shape(value)
//     val splitShape = Basic.concatenate(Seq(batchSize(NewAxis), beamWidth(NewAxis), valueShape(1 ::)), axis = 0)
//     Basic.reshape(value, splitShape)
//   }

//   /** Adds a new dimension to `value` and tiles it along that dimension, `beamWidth` times.
//     *
//     * If `value` is a scalar, then a vector is returned, otherwise, the new beam dimension is always introduced as the
//     * second dimension. For example, if `value` has shape `[batchSize, ...]`, then the resulting tensor will have shape
//     * `[batchSize, beamWidth]`.
//     *
//     * @param  value     Tensor to tile.
//     * @param  beamWidth Beam width (i.e., number of times to tile value).
//     * @return `value` tiled and reshaped so that it now has a beam dimension.
//     * @throws InvalidArgumentException If the rank of `value` is unknown.
//     */
//   @throws[InvalidArgumentException]
//   private[BeamSearchDecoder2] def tileForBeamSearch[T: TF](
//       value: Output[T],
//       beamWidth: Int
//   ): Output[T] = {
//     if (value.rank == -1) {
//       throw InvalidArgumentException("The provided tensor must have statically known rank.")
//     } else if (value.rank == 0) {
//       Basic.tile(value.expandDims(0), multiples = Tensor(beamWidth))
//     } else {
//       val multiples = ArrayBuffer.fill(value.rank + 1)(1)
//       multiples(1) = beamWidth
//       Basic.tile(value.expandDims(1), multiples)
//     }
//   }

//   /** Returns the shape of `value`, but with all its inner dimensions replaced with `-1`.
//     *
//     * @param  value Tensor whose shape invariants to return.
//     * @return `value` shape with all its inner dimensions replaced with `-1`.
//     */
//   private[BeamSearchDecoder2] def getStateShapeInvariants(
//       value: Output[_]
//   ): Shape = {
//     Shape.fromSeq(value.shape.asArray.zipWithIndex.map(s => {
//       if (s._2 == 0 || s._2 == value.rank - 1)
//         s._1
//       else
//         -1
//     }))
//   }

//   /** Beam search decoder state.
//     *
//     * @param  sequences     Tensor of sequences we need to gather from, with shape `[batchSize, beamWidth, length]`.
//     * @param  scores        Tensor of scores for each sequence in `sequences`, with shape `[batchSize, beamWidth]`.
//     * @param  finished      Tensor of boolean flags indicating whether each sequence in `sequences` has finished
//     *                       (i.e. has reached end-of-sequence), with shape `[batchSize, beamWidth]`.
//     * @param  decodingState Decoding state.
//     */
//   case class BeamSearchState[T: TF : IsReal, S: OutputStructure](
//       sequences: Output[T],
//       scores: Output[Float],
//       finished: Output[Boolean],
//       decodingState: Option[S])

//   /** Given a beam search decoder state, this method will gather the top `beamWidth` sequences, scores, etc., and
//     * return a new, updated beam search decoder state.
//     *
//     * This method permits easy introspection using the TensorFlow debugger. It add named ops, prefixed by `name` and
//     * with names `TopKSequences`, `TopKFinished`, and `TopKScores`..
//     *
//     * @param  state           Beam search state.
//     * @param  scoresToGather  Tensor of scores to gather, for each sequence in `sequences`, with shape
//     *                         `[batchSize, beamWidth]`. These scores may differ from `scores` because for `growAlive`,
//     *                         we need to return log probabilities, while for `growFinished`, we need to return length
//     *                         penalized scores.
//     * @param  beamWidth       Integer specifying the beam width.
//     * @param  useTPU          Boolean flag indicating whether the created ops will be executed on TPUs.
//     * @param  name            Name scope to use for the created ops.
//     */
//   private[BeamSearchDecoder2] def computeTopKScoresAndSequences[T: TF : IsReal, S: OutputStructure](
//       state: BeamSearchState[T, S],
//       scoresToGather: Output[Float],
//       beamWidth: Int,
//       useTPU: Boolean = false,
//       name: String = "ComputeTopKScoresAndSequences"
//   ): BeamSearchState[T, S] = {
//     Op.nameScope(name) {
//       val batchSize = Basic.shape(state.sequences).slice(0)
//       if (useTPU) {
//         val topKIndices = computeTopKUnique(state.scores, k=beamWidth)._2
//         // Gather up the highest scoring sequences. We give each operation added a concrete name to simplify observing
//         // these operations with the TensorFlow debugger. Clients can capture these tensors by watching these node names.
//         val topKSequences = tpuGather(state.sequences, topKIndices, name = "TopKSequences")
//         val topKFinished = tpuGather(state.finished, topKIndices, name = "TopKFinished")
//         val topKScores = tpuGather(scoresToGather, topKIndices, name = "TopKScores")
//         val topKStates = state.decodingState.map(s => {
//           OutputStructure[S].map(s, new OutputStructure.Converter {
//               override def apply[T](value: Output[T], shape: Option[Shape]): Output[T] = {
//                 tpuGather(value, topKIndices, name = "TopKStates")
//               }
//           })
//         })

//         BeamSearchState(topKSequences, topKScores, topKFinished, topKStates)
//       } else {
//         val topKIndices = NN.topK(state.scores, k = beamWidth)

//         // The next steps are used to create indices for `gatherND` to pull out the top `k` sequences from
//         // sequences based on scores. The batch indices tensor looks like `[ [0, 0, 0, ...], [1, 1, 1, ...], ... ]`,
//         // and it indicates which batch each beam item is in. This will create the `i` of the `[i, j]` coordinate
//         // needed for a gather.
//         val batchIndices = computeBatchIndices(batchSize, beamWidth)

//         // The top indices specify which indices to use for the gather. The result of the stacking operation is a tensor
//         // with shape `[batchSize, beamWidth, 2]`, where the last dimension specifies the gather indices.
//         val topIndices = Basic.stack(Seq(batchIndices, topKIndices), axis = 2)

//         // Next, we gather the highest scoring sequences. We give each operation added a concrete name to simplify
//         // observing these operations with the TensorFlow debugger. Clients can capture these tensors by watching these
//         // node names.
//         val topKSequences = Basic.gatherND(state.sequences, topIndices, name = "TopKSequences")
//         val topKFinished = Basic.gatherND(state.finished, topIndices, name = "TopKFinished")
//         val topKScores = Basic.gatherND(scoresToGather, topIndices, name = "TopKScores")
//         val topKStates = state.decodingState.map(s => {
//           OutputStructure[S].map(s, new OutputStructure.Converter {
//               override def apply[T](value: Output[T], shape: Option[Shape]): Output[T] = {
//                 Basic.gatherND(value, topIndices, name = "TopKStates")
//               }
//           })
//         })

//         BeamSearchState(topKSequences, topKScores, topKFinished, topKStates)
//       }
//     }
//   }

//   /** Computes the `i`th coordinate that contains the batch index for gathers.
//     *
//     * The batch indices tensor looks like `[ [0, 0, 0, ...], [1, 1, 1, ...], ... ]`, and it indicates which batch each
//     * beam item is in. This will create the `i` of the `[i, j]` coordinate needed for a gather.
//     *
//     * @param  batchSize Batch size.
//     * @param  beamWidth Beam width.
//     * @return Tensor with shape `[batchSize, beamWidth]`, containing batch indices.
//     */
//   private[BeamSearchDecoder2] def computeBatchIndices(
//       batchSize: Output[Int],
//       beamWidth: Output[Int]
//   ): Output[Int] = {
//     val batchIndices = Math.truncateDivide(Math.range(0, batchSize * beamWidth), beamWidth)
//     Basic.reshape(batchIndices, Seq(batchSize, beamWidth))
//   }

//   /** Finds the values and indices of the `k` largest entries in `input`.
//     *
//     * Instead of performing a sort like the `topK` op, this method finds the maximum value `k` times. The running time
//     * is proportional to `k`, which is going to be faster for small values of `k`. The current implementation only
//     * supports inputs of rank 2. In addition, iota is used to replace the lower bits of each element, which makes the
//     * selection more stable when there are equal elements. The overhead is that output values are approximate.
//     *
//     * @param  input Input tensor with shape `[batchSize, depth]`.
//     * @param  k     Number of top elements to select.
//     * @return Tuple containing:
//     *           - Tensor with shape `[batchSize, k]`, which contains the top `k` elements in `input`.
//     *           - Tensor with shape `[batchSize, k]`, which contains the indices of the top `k` elements in `input`.
//     */
//   private[BeamSearchDecoder2] def computeTopKUnique[T: TF](
//       input: Output[T],
//       k: Int
//   ): (Output[T], Output[Int]) = {
//     val uniqueInput = makeUnique(input.toFloat)
//     val height = uniqueInput.shape(0)
//     val width = uniqueInput.shape(1)
//     val negInfR0 = Basic.constant[Float](Float.NegativeInfinity)
//     val negInfR2 = Basic.fill(Shape(height, width))(negInfR0)
//     val infFilteredInput = Math.select(Math.isNaN(uniqueInput), negInfR2, uniqueInput)

//     // Select the current largest value `k` times and keep the selected values in `topKR2`.
//     // The selected largest values are marked as the smallest values to avoid being selected again.
//     var temp = uniqueInput
//     var topKR2 = Basic.zeros[Float](Shape(height, k))
//     for (i <- 0 until k) {
//       val kthOrderStatistics = Math.max(temp, axes = 1, keepDims = true)
//       val kMask = Basic.tile(
//         input = Math.equal[Int](Math.range[Int](0, k), Basic.fill(Shape(k))(Basic.constant[Int](i))).expandDims(0),
//         multiples = Tensor[Int](height, 1))
//       topKR2 = Math.select(kMask, Basic.tile(kthOrderStatistic, Tensor[Int](1, k)), topKR2)
//       temp = Math.select(
//         condition = Math.greaterEqual(uniqueInput, Basic.tile(kthOrderStatistic, Tensor[Int](1, width))),
//         x = negInfR2,
//         y = uniqueInput)
//     }

//     val log2Ceil = scala.math.ceil(scala.math.log(width) / scala.math.log(2)).toInt
//     val nextPowerOfTwo = 1 << log2Ceil
//     val countMask = nextPowerOfTwo - 1 // TODO: Why is this not the same as the one in `makeUnique`.
//     val countMaskR0 = Basic.constant(countMask)
//     val countMaskR2 = Basic.fill(Shape(height, k))(countMaskR0)
//     val topKR2Int = topKR2.bitcastTo[Int]
//     val topKIndicesR2 = Math.bitwise.and(topKR2Int, countMaskR2)
//     (topKR2.castTo[T], topKIndicesR2)
//   }

//   /** Replaces the lower bits of each element with iota.
//     *
//     * The iota is used to derive the index, and also serves the purpose of making each element unique to break ties.
//     *
//     * @param  input Input tensor with rank 2.
//     * @return Resulting tensor after an element-wise transformation on `input`.
//     * @throws InvalidShapeException If the rank of `input` is not equal to 2.
//     */
//   @throws[InvalidShapeException]
//   private[BeamSearchDecoder2] def makeUnique(
//       input: Output[Float]
//   ): Output[Float] = {
//     if (input.rank != 2)
//       throw InvalidShapeException(s"Expected rank-2 input, but got input with shape ${input.shape}.")

//     val height = input.shape(0)
//     val width = input.shape(1)

//     // The count mask is used to mask away the low order bits thus ensuring that every element is distinct.
//     val log2Ceil = scala.math.ceil(scala.math.log(width) / scala.math.log(2)).toInt
//     val nextPowerOfTwo = 1 << log2Ceil
//     val countMask = ~(nextPowerOfTwo - 1)
//     val countMaskR0 = Basic.constant(countMask)
//     val countMaskR2 = Basic.fill(Shape(height, width))(countMaskR0)

//     // This is the bit representation of the smallest positive normal floating point number. The sign is zero, the
//     // exponent is one, and the fraction is zero.
//     val smallestNormal = 1 << 23
//     val smallestNormalR0 = Basic.constant(smallestNormal)
//     val smallestNormalR2 = Basic.fill(Shape(height, width))(smallestNormalR0)

//     // This is used to mask away the sign bit when computing the absolute value.
//     val lowBitMask = ~(1 << 31)
//     val lowBitMaskR0 = Basic.constant(lowBitMask)
//     val lowBitMaskR2 = Basic.fill(Shape(height, width))(lowBitMaskR0)

//     // Compare the absolute value with positive zero to handle negative zero.
//     val inputR2 = input.bitcastTo[Int]
//     val absR2 = Math.bitwise.and(inputR2, lowBitMaskR2)
//     val inputNoZerosR2 = Math.select(
//       condition = Math.equal(absR2, 0),
//       x = Math.bitwise.or(inputR2, smallestNormalR2),
//       y = inputR2)

//     val iota = Basic.tile(
//       input = Math.range[Int](0, width).expandDims(0),
//       multiples = Tensor(height, 1))

//     // Discard the low-order bits and replace with iota.
//     val andR2 = Math.bitwise.and(inputNoZerosR2, countMaskR2)
//     val orR2 = Math.bitwise.or(andR2, iota)
//     orR2.bitcastTo[Float]
//   }

//   /** Fast gather implementation for models running on TPUs.
//     *
//     * This function uses a one-hot tensor and a batch matrix multiplications to perform the gather, which is faster
//     * than `gatherND` on TPUs. For integer inputs, (i.e., tensors to gather from), `batchGather` is used instead, in
//     * order to maintain reasonable accuracy.
//     *
//     * @param  input   Tensor from which to gather values, with shape `[batchSize, ...]`.
//     * @param  indices Indices to use for the gather, which is a tensor with shape `[batchSize, numIndices]`.
//     * @param  name    Namescope for the created ops.
//     * @return Tensor that contains the gathered values and has the same rank as `input`, and shape
//     *         `[batchSize, numIndices, ...]`.
//     */
//   private[BeamSearchDecoder2] def tpuGather[T: TF : IsNotQuantized, I: TF : IsIntOrLong](
//       input: Output[T],
//       indices: Output[I],
//       name: String = "Gather"
//   ): Output[T] = {
//     Op.nameScope(name) {
//       // If the data type is `INT32`, use the gather instead of the ont-hot matrix multiplication, in order to avoid
//       // precision loss. The maximum integer value that can be represented by `BFLOAT16` in the MXU is 256, which is
//       // smaller than the possible `indices` values. Encoding/decoding can potentially be used to make it work, but
//       // the benefit is small right now.
//       if (input.dataType == INT32) {
//         Basic.batchGather(input, indices)
//       } else if (input.dataType != FLOAT32) {
//         val floatResult = tpuGather(input.toFloat, indices)
//         floatResult.castTo[T]
//       } else {
//         val inputShape = Basic.shape(input)
//         val inputRank = input.rank
//         // Adjust the shape of the input to match one-hot indices,
//         // which is a requirement for batch matrix multiplication.
//         val reshapedInput = inputRank match {
//           case 2 => Basic.expandDims(input, axis = -1)
//           case r if r > 3 => Basic.reshape(input, Seq(inputShape(0), inputShape(1), Basic.constant[Int](-1)))
//           case _ => input
//         }
//         val oneHot = Basic.oneHot[T, I](indices, depth = inputShape(1).toInt)
//         val gatherResult = Math.matmul(oneHot, reshapedInput)
//         inputRank match {
//           case 2 => Basic.squeeze(gatherResult, axes = Seq(-1))
//           case r if r > 3 =>
//             val resultShape = Basic.concatenate(Seq(
//               inputShape(0).expandDims(0),
//               Basic.shape(indices).slice(1).expandDims(0),
//               inputShape(2 ::)
//             ), axis = 0)
//             Basic.reshape(gatherResult, resultShape)
//           case _ => gatherResult
//         }
//       }
//     }
//   }
// }
