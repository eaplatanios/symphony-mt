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

package org.platanios.symphony.mt.models.helpers.decoders

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.exception.{InvalidArgumentException, InvalidShapeException}
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape, Zero}

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer
import scala.language.postfixOps

// TODO: [SEQ2SEQ] Abstract away the log-softmax/scoring function.

/** Recurrent Neural Network (RNN) that uses beam search to find the highest scoring sequence (i.e., perform decoding).
  *
  * @param  cell                RNN cell to use for decoding.
  * @param  initialCellState    Initial RNN cell state to use for starting the decoding process.
  * @param  embeddingFn         Function that takes an vector of IDs and returns the corresponding embedded
  *                             values that will be passed to the decoder input.
  * @param  beginTokens         Vector with length equal to the batch size, which contains the begin-of-sequence
  *                             token IDs.
  * @param  endToken            Scalar containing the end-of-sequence token ID (i.e., token ID which marks the end of
  *                             decoding).
  * @param  beamWidth           Beam width to use for the beam search while decoding.
  * @param  lengthPenalty       Length penalty method.
  * @param  outputLayer         Output layer to use that is applied at the outputs of the provided RNN cell before
  *                             returning them.
  * @param  reorderTensorArrays If `true`, `TensorArray`s' elements within the cell state will be reordered according to
  *                             the beam search path. If the `TensorArray` can be reordered, the stacked form will be
  *                             returned. Otherwise, the `TensorArray` will be returned as is. Set this flag to `false`
  *                             if the cell state contains any `TensorArray`s that are not amenable to reordering.
  * @param  name                Name prefix used for all created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class BeamSearchDecoder[T: TF, State: OutputStructure, StateShape](
    override val cell: tf.RNNCell[Output[T], State, Shape, StateShape],
    val initialCellState: State,
    val embeddingFn: Output[Int] => Output[T],
    val beginTokens: Output[Int],
    val endToken: Output[Int],
    val beamWidth: Int,
    val lengthPenalty: LengthPenalty = NoPenalty,
    val outputLayer: Output[T] => Output[T] = (o: Output[T]) => o,
    val reorderTensorArrays: Boolean = true,
    override val name: String = "BeamSearchRNNDecoder"
)(implicit
    evOutputToShapeState: OutputToShape.Aux[State, StateShape]
) extends Decoder[
    /* Out           */ Output[T],
    /* State         */ State,
    /* DecOut        */ BeamSearchDecoder.BeamSearchDecoderOutput,
    /* DecState      */ BeamSearchDecoder.BeamSearchDecoderState[State],
    /* DecFinalOut   */ BeamSearchDecoder.BeamSearchFinalOutput,
    /* DecFinalState */ BeamSearchDecoder.BeamSearchDecoderState[State],
    /* Shapes        */ Shape, StateShape, (Shape, Shape, Shape), (StateShape, Shape, Shape, Shape)
    ](cell, name) {
  if (beginTokens.rank != 1)
    throw InvalidShapeException(s"'beginTokens' (shape = ${beginTokens.shape}) must have rank 1.")
  if (endToken.rank != 0)
    throw InvalidShapeException(s"'endToken' (shape = ${endToken.shape}) must have rank 0.")

  /** Scalar tensor representing the batch size of the input values. */
  override val batchSize: Output[Int] = {
    tf.nameScope(name) {
      tf.size(beginTokens).castTo[Int]
    }
  }

  private val beginInput: Output[T] = {
    tf.nameScope(name) {
      val tiledBeginTokens = tf.tile(tf.expandDims(beginTokens, 1), tf.stack[Int](Seq(1, beamWidth)))
      embeddingFn(tiledBeginTokens)
    }
  }

  /** Describes whether the decoder keeps track of finished states.
    *
    * Most decoders will emit a true/false `finished` value independently at each time step. In this case, the
    * `dynamicDecode()` function keeps track of which batch entries have already finished, and performs a logical OR to
    * insert new batches to the finished set.
    *
    * Some decoders, however, shuffle batches/beams between time steps and `dynamicDecode()` will mix up the finished
    * state across these entries because it does not track the reshuffling across time steps. In this case, it is up to
    * the decoder to declare that it will keep track of its own finished state by setting this property to `true`.
    *
    * The beam-search decoder shuffles its beams and their finished state. For this reason, it conflicts with the
    * `dynamicDecode` function's tracking of finished states. Setting this property to `true` avoids early stopping of
    * decoding due to mismanagement of the finished state in `dynamicDecode`.
    */
  override val tracksOwnFinished: Boolean = {
    true
  }

  /** Returns a zero-valued output for this decoder. */
  override def zeroOutput: BeamSearchDecoder.BeamSearchDecoderOutput = {
    // We assume the data type of the cell is the same as the initial cell state's first component tensor data type.
    val zScores = Zero[Output[Float]].zero(batchSize, Shape(beamWidth), "ZeroScores")
    val zPredictedIDs = Zero[Output[Int]].zero(batchSize, Shape(beamWidth), "ZeroPredictedIDs")
    val zParentIDs = Zero[Output[Int]].zero(batchSize, Shape(beamWidth), "ZeroParentIDs")
    BeamSearchDecoder.BeamSearchDecoderOutput(zScores, zPredictedIDs, zParentIDs)
  }

  /** This method is called before any decoding iterations. It computes the initial input values and the initial state.
    *
    * @return Tuple containing: (i) a scalar tensor specifying whether initialization has finished,
    *         (ii) the next input, and (iii) the initial decoder state.
    */
  override def initialize(): (Output[Boolean], Output[T], BeamSearchDecoder.BeamSearchDecoderState[State]) = {
    tf.nameScope(s"$name/Initialize") {
      val tiledInitialCellState = {
        val evStructureState = OutputStructure[State]
        if (evStructureState.outputs(initialCellState).exists(_.rank == -1))
          throw InvalidArgumentException("All tensors in the state need to have known rank for the beam search decoder.")
        evStructureState.map(initialCellState, new OutputStructure.Converter {
          @throws[InvalidArgumentException]
          override def apply[V](value: Output[V], shape: Option[Shape]): Output[V] = {
            implicit val evTF: TF[V] = TF.fromDataType(value.dataType)
            if (value.rank == -1) {
              throw InvalidArgumentException("The provided tensor must have statically known rank.")
            } else if (value.rank == 0) {
              val tiling = Tensor(beamWidth)
              val tiled = tf.tile(value.expandDims(0), tiling)
              tiled.setShape(Shape(beamWidth))
              tiled
            } else {
              val tiling = ArrayBuffer.fill(value.rank + 1)(1)
              tiling(1) = beamWidth
              tf.tile(value.expandDims(1), tiling)
            }
          }
        })
      }
      val finished = tf.oneHot[Boolean, Int](
        indices = tf.zeros[Int, Int](batchSize.expandDims(0)),
        depth = beamWidth,
        onValue = false,
        offValue = true)
      val initialState = BeamSearchDecoder.BeamSearchDecoderState[State](
        modelState = tiledInitialCellState,
        logProbabilities = tf.oneHot[Float, Int](
          indices = tf.zeros[Int](batchSize.expandDims(0)),
          depth = beamWidth,
          onValue = tf.zeros[Float](Shape()),
          offValue = tf.constant(Float.MinValue)),
        finished = finished,
        sequenceLengths = tf.zeros[Int](tf.stack[Int](Seq(batchSize, beamWidth))))
      (finished, beginInput, initialState)
    }
  }

  /** This method specifies what happens in each step of decoding.
    *
    * @return Tuple containing: (i) the decoder output, (ii) the next state, (iii) the next inputs, and (iv) a scalar
    *         tensor specifying whether sampling has finished.
    */
  override def next(
      time: Output[Int],
      input: Output[T],
      state: BeamSearchDecoder.BeamSearchDecoderState[State]
  ): (BeamSearchDecoder.BeamSearchDecoderOutput, BeamSearchDecoder.BeamSearchDecoderState[State], Output[T], Output[Boolean]) = {
    val evStructureState = OutputStructure[State]
    tf.nameScope(s"$name/Next") {
      val mergedInput = BeamSearchDecoder.MergeBatchBeamsConverter(batchSize, beamWidth)(input, Some(input.shape(2 ::)))
      val mergedCellState = evOutputToShapeState.map(
        state.modelState, Some(cell.stateShape),
        BeamSearchDecoder.MaybeTensorConverter(
          BeamSearchDecoder.MergeBatchBeamsConverter(batchSize, beamWidth)))
      val mergedNextTuple = cell(tf.RNNTuple(mergedInput, mergedCellState))
      val nextTupleOutput = outputLayer(BeamSearchDecoder.SplitBatchBeamsConverter(batchSize, beamWidth)(
        mergedNextTuple.output, Some(mergedNextTuple.output.shape(1 ::))))
      val nextTupleState = evOutputToShapeState.map(
        mergedNextTuple.state, Some(cell.stateShape),
        BeamSearchDecoder.MaybeTensorConverter(
          BeamSearchDecoder.SplitBatchBeamsConverter(batchSize, beamWidth)))

      // Perform the beam search step.
      val staticBatchSize = Output.constantValue(batchSize).map(_.scalar).getOrElse(-1)

      // Calculate the current lengths of the predictions.
      val predictionLengths = state.sequenceLengths
      val previouslyFinished = state.finished

      // Calculate the total log probabilities for the new hypotheses (final shape = [batchSize, beamWidth, vocabSize]).
      val expandedStateLogProbabilities = tf.expandDims(
        state.logProbabilities, axis = 2, name = "LogProbabilities/ExpandDims")
      val stepLogProbabilities = BeamSearchDecoder.maskLogProbabilities(
        tf.logSoftmax(nextTupleOutput.toFloat), endToken, previouslyFinished)
      val totalLogProbabilities = expandedStateLogProbabilities + stepLogProbabilities

      // Calculate the continuation lengths by adding to all continuing search states.
      val vocabSize = {
        if (nextTupleOutput.shape(-1) != -1)
          tf.constant(nextTupleOutput.shape(-1))
        else
          tf.shape(nextTupleOutput).toInt.slice(-1)
      }

      val lengthsToAdd = tf.oneHot[Int, Int](
        indices = tf.fill[Int, Int](tf.stack[Int](Seq(batchSize, beamWidth)))(endToken),
        depth = vocabSize,
        onValue = 0,
        offValue = 1)
      val addMask = tf.logicalNot(previouslyFinished).toInt
      val newPredictionLengths = tf.add(
        lengthsToAdd * tf.expandDims(addMask, axis = 2, name = "AddMask/ExpandDims"),
        tf.expandDims(predictionLengths, axis = 2, name = "PredictionLengths/ExpandDims"))

      // Calculate the scores for each search state.
      val scores = lengthPenalty(totalLogProbabilities, newPredictionLengths).toFloat

      // During the first time step we only consider the initial search state.
      val scoresFlat = tf.reshape(scores, tf.stack[Int](Seq(batchSize, -1)))

      // Pick the next search states according to the specified successors function.
      val nextBeamSize = tf.constant(beamWidth, name = "BeamWidth")
      val (nextBeamScores, wordIndices) = tf.topK(scoresFlat, nextBeamSize)
      nextBeamScores.setShape(Shape(staticBatchSize, beamWidth))
      wordIndices.setShape(Shape(staticBatchSize, beamWidth))

      // Pick out the log probabilities, search state indices, and states according to the chosen predictions.
      val nextBeamLogProbabilities = BeamSearchDecoder.gather(
        gatherIndices = wordIndices,
        gatherFrom = totalLogProbabilities,
        batchSize = batchSize,
        rangeSize = vocabSize * beamWidth,
        gatherShape = Seq(-1),
        name = "NextBeamLogProbabilities")
      val nextPredictedIDs = tf.mod(wordIndices, vocabSize, name = "NextBeamPredictedIDs").toInt
      val nextParentIDs = tf.divide(wordIndices, vocabSize, name = "NextBeamParentIDs").toInt

      // Append the new IDs to the current predictions.
      val gatheredFinished = BeamSearchDecoder.gather(
        gatherIndices = nextParentIDs,
        gatherFrom = previouslyFinished,
        batchSize = batchSize,
        rangeSize = beamWidth,
        gatherShape = Seq(-1),
        name = "NextBeamFinishedGather")

      val nextFinished = tf.logicalOr(
        gatheredFinished, tf.equal(nextPredictedIDs, endToken),
        name = "NextBeamFinished")

      // Calculate the length of the next predictions:
      //   1. Finished search states remain unchanged.
      //   2. Search states that just finished (i.e., `endToken` predicted) have their length increased by 1.
      //   3. Search states that have not yet finished have their length increased by 1.
      val nextPredictionLengths = BeamSearchDecoder.gather(
        gatherIndices = nextParentIDs,
        gatherFrom = state.sequenceLengths,
        batchSize = batchSize,
        rangeSize = beamWidth,
        gatherShape = Seq(-1),
        name = "NextBeamLengthsGather"
      ) + tf.logicalNot(gatheredFinished).toInt

      // Pick out the cell state according to the next search state parent IDs. We use a different gather shape here
      // because the cell state tensors (i.e., the tensors that would be gathered from) all have rank greater than two
      // and we need to preserve those dimensions.
      val gatheredNextTupleState = evStructureState.map(
        nextTupleState,
        BeamSearchDecoder.MaybeTensorConverter(
          new OutputStructure.Converter {
            override def apply[V](value: Output[V], shape: Option[Shape]): Output[V] = {
              implicit val evVTF: TF[V] = TF.fromDataType(value.dataType)
              val valueShape = tf.shape(value)
              val gatherShape = (batchSize * beamWidth) +: (2 until value.rank).map(valueShape(_))
              BeamSearchDecoder.gather(
                nextParentIDs, value, batchSize, beamWidth, gatherShape, name = "NextBeamStateGather")
            }
          }))

      val nextState = BeamSearchDecoder.BeamSearchDecoderState[State](
        gatheredNextTupleState, nextBeamLogProbabilities, nextPredictionLengths, nextFinished)
      val output = BeamSearchDecoder.BeamSearchDecoderOutput(nextBeamScores, nextPredictedIDs, nextParentIDs)

      val nextInput = tf.cond(
        nextFinished.all(),
        () => beginInput,
        () => embeddingFn(nextPredictedIDs))

      (output, nextState, nextInput, nextFinished)
    }
  }

  /** Finalizes the output of the decoding process.
    *
    * @param  output Final output after decoding.
    * @param  state  Final state after decoding.
    * @return Finalized output and state to return from the decoding process.
    */
  override def finalize(
      output: BeamSearchDecoder.BeamSearchDecoderOutput,
      state: BeamSearchDecoder.BeamSearchDecoderState[State],
      sequenceLengths: Output[Int]
  ): (BeamSearchDecoder.BeamSearchFinalOutput, BeamSearchDecoder.BeamSearchDecoderState[State], Output[Int]) = {
    // Get the maximum sequence length across all search states for each batch
    val maxSequenceLengths = state.sequenceLengths.max(Tensor[Int](1))
    val predictedIDs = BeamSearchDecoder.gatherTree(
      output.predictedIDs, output.parentIDs, maxSequenceLengths, endToken)
    val finalOutput = BeamSearchDecoder.BeamSearchFinalOutput(predictedIDs, output)
    var finalState = state
    if (reorderTensorArrays)
      finalState = state.copy[State](modelState = OutputStructure[State].map(
        state.modelState,
        BeamSearchDecoder.MaybeSortTensorArrayBeamsConverter(
          state.sequenceLengths, output.parentIDs, batchSize, beamWidth)))
    (finalOutput, finalState, finalState.sequenceLengths)
  }
}

object BeamSearchDecoder {
  protected[BeamSearchDecoder] val logger = Logger(LoggerFactory.getLogger("Ops / Beam Search Decoder"))

  def apply[T: TF, State: OutputStructure, StateShape](
      cell: tf.RNNCell[Output[T], State, Shape, StateShape],
      initialCellState: State,
      embeddingFn: Output[Int] => Output[T],
      beginTokens: Output[Int],
      endToken: Output[Int],
      beamWidth: Int,
      lengthPenalty: LengthPenalty = NoPenalty,
      outputLayer: Output[T] => Output[T] = (o: Output[T]) => o,
      reorderTensorArrays: Boolean = true,
      name: String = "BeamSearchRNNDecoder"
  )(implicit
      evOutputToShapeState: OutputToShape.Aux[State, StateShape]
  ): BeamSearchDecoder[T, State, StateShape] = {
    new BeamSearchDecoder[T, State, StateShape](
      cell, initialCellState, embeddingFn, beginTokens, endToken,
      beamWidth, lengthPenalty, outputLayer, reorderTensorArrays, name)
  }

  case class BeamSearchDecoderOutput(
      scores: Output[Float],
      predictedIDs: Output[Int],
      parentIDs: Output[Int])

  case class BeamSearchDecoderState[State](
      modelState: State,
      logProbabilities: Output[Float],
      sequenceLengths: Output[Int],
      finished: Output[Boolean])

  /** Final outputs returned by the beam search after all decoding is finished.
    *
    * @param  predictedIDs Tensor of shape `[batchSize, T, beamWidth]` (or `[T, batchSize, beamWidth]`,
    *                      if `outputTimeMajor == true`) containing the final prediction IDs. The search states are
    *                      ordered from best to worst.
    * @param  output       State of the beam search at the end of decoding.
    */
  case class BeamSearchFinalOutput(
      predictedIDs: Output[Int],
      output: BeamSearchDecoderOutput)

  /** Masks log probabilities. The result is that finished search states allocate all probability mass to `endToken` and
    * unfinished search states remain unchanged.
    *
    * @param  logProbabilities Log probability for each hypothesis, which is a tensor with shape
    *                          `[batchSize, beamWidth, vocabSize]`.
    * @param  endToken         Scalar tensor containing the end-of-sequence token ID.
    * @param  finished         Tensor of shape `[batchSize, beamWidth]` that specifies which elements in the beam have
    *                          finished decoding.
    * @return Tensor of shape `[batchSize, beamWidth, vocabSize]`, where unfinished search states stay unchanged and
    *         finished search states are replaced with a tensor with all probability mass allocated to `endToken`.
    */
  private[BeamSearchDecoder] def maskLogProbabilities(
      logProbabilities: Output[Float],
      endToken: Output[Int],
      finished: Output[Boolean]
  ): Output[Float] = {
    val vocabSize = tf.shape(logProbabilities).castTo[Int].slice(2)
    // Finished examples are replaced with a vector that has all its probability mass on `endToken`
    val finishedRow = tf.oneHot[Float, Int](
      indices = endToken,
      depth = vocabSize,
      onValue = tf.zeros[Float](Shape()),
      offValue = tf.constant(Float.MinValue))
    val finishedLogProbabilities = tf.tile(
      input = finishedRow.reshape(Shape(1, 1, -1)),
      multiples = tf.concatenate[Int](Seq(tf.shape(finished).castTo[Int], Tensor(1)), 0))
    val finishedMask = tf.tile(
      input = tf.expandDims(finished, axis = 2, name = "Finished/ExpandDims"),
      multiples = tf.stack[Int](Seq(1, 1, vocabSize)))
    tf.select(finishedMask, finishedLogProbabilities, logProbabilities)
  }

  /** The `gatherTree` op calculates the full search states from the per-step IDs and parent beam IDs.
    *
    * On a CPU, if an out-of-bounds parent ID is found, an error is returned. On a GPU, if an out-of-bounds parent ID
    * is found, a `-1` is stored in the corresponding output value and the execution for that beam returns early.
    *
    * For a given beam, past the time step containing the first decoded `endToken`, all values are filled in with
    * `endToken`.
    *
    * @param  stepIDs            Tensor with shape `[maxTime, batchSize, beamWidth]`, containing the step IDs.
    * @param  parentIDs          Tensor with shape `[maxTime, batchSize, beamWidth]`, containing the parent IDs.
    * @param  maxSequenceLengths Tensor with shape `[batchSize]`, containing the sequence lengths.
    * @param  endToken           Scalar tensor containing the end-of-sequence token ID.
    * @param  name               Name for the created op.
    * @return Created op output.
    */
  private[decoders] def gatherTree(
      stepIDs: Output[Int],
      parentIDs: Output[Int],
      maxSequenceLengths: Output[Int],
      endToken: Output[Int],
      name: String = "GatherTree"
  ): Output[Int] = {
    Op.Builder[(Output[Int], Output[Int], Output[Int], Output[Int]), Output[Int]](
      opType = "GatherTree",
      name = name,
      input = (stepIDs, parentIDs, maxSequenceLengths, endToken)
    ).build().output
  }

  /** Maybe converts the provided tensor. */
  private[BeamSearchDecoder] case class MaybeTensorConverter(
      innerConverter: OutputStructure.Converter
  ) extends OutputStructure.Converter {
    @throws[InvalidArgumentException]
    override def apply[T](value: Output[T], shape: Option[Shape]): Output[T] = {
      if (value.rank == -1) {
        throw InvalidArgumentException(s"Expected tensor ($value) to have known rank, but it was unknown.")
      } else if (value.rank == 0) {
        value
      } else {
        innerConverter(value, shape)
      }
    }

    @throws[InvalidArgumentException]
    override def apply[T](value: tf.data.Dataset[T], shape: Option[Shape]): tf.data.Dataset[T] = {
      throw InvalidArgumentException("Unsupported argument type for use with the beam search decoder.")
    }
  }

  /** Converts the provided tensor structure from batches by beams into batches of beams, by merging them accordingly.
    *
    * More precisely, `value` consists of tensors with shape `[batchSize * beamWidth] ++ ...` and this method reshapes
    * them into tensors with shape `[batchSize, beamWidth] ++ ...`.
    *
    * @throws InvalidArgumentException If `value` is of an unsupported type.
    * @throws InvalidShapeException    If the provided value contains any tensors of unknown rank, or if, after
    *                                  reshaping, the new tensor is not shaped `[batchSize, beamWidth] ++ ...`
    *                                  (assuming that both `batchSize` and `beamWidth` are known statically).
    */
  private[BeamSearchDecoder] case class SplitBatchBeamsConverter(
      batchSize: Output[Int],
      beamWidth: Int
  ) extends OutputStructure.Converter {
    @throws[InvalidArgumentException]
    @throws[InvalidShapeException]
    override def apply[T](value: Output[T], shape: Option[Shape]): Output[T] = {
      implicit val evTTF: TF[T] = TF.fromDataType(value.dataType)
      if (shape.isDefined && shape.get.rank == 0) {
        value
      } else {
        val valueShape = tf.shape(value)
        val reshapedValue = tf.reshape(value, tf.concatenate(Seq(
          batchSize.expandDims(0),
          Output[Int](beamWidth),
          valueShape(1 ::)), axis = 0))
        val staticBatchSize = Output.constantValue(batchSize).map(_.scalar).getOrElse(-1)
        val expectedReshapedShape = Shape(staticBatchSize, beamWidth) ++ shape.get
        if (!reshapedValue.shape.isCompatibleWith(expectedReshapedShape)) {
          throw InvalidShapeException(
            "Unexpected behavior when reshaping between beam width and batch size. " +
                s"The reshaped tensor has shape: ${reshapedValue.shape}. " +
                s"We expected it to have shape [batchSize, beamWidth, depth] == $expectedReshapedShape. " +
                "Perhaps you forgot to create a zero state with batchSize = encoderBatchSize * beamWidth?")
        }
        reshapedValue.setShape(expectedReshapedShape)
        reshapedValue
      }
    }

    @throws[InvalidArgumentException]
    override def apply[T](value: tf.data.Dataset[T], shape: Option[Shape]): tf.data.Dataset[T] = {
      throw InvalidArgumentException("Unsupported argument type for use with the beam search decoder.")
    }
  }

  /** Converts the provided tensor structure from a batch of search states into a batch by beams, by merging them
    * accordingly>
    *
    * More precisely, `value` consists of tensors with shape `[batchSize, beamWidth] ++ ...` and this method reshapes
    * them into tensors with shape `[batchSize * beamWidth] ++ ...`.
    *
    * @throws InvalidArgumentException If `value` is of an unsupported type.
    * @throws InvalidShapeException    If the provided value contains any tensors of unknown rank, or if, after
    *                                  reshaping, the new tensor is not shaped `[batchSize * beamWidth] ++ ...`
    *                                  (assuming that both `batchSize` and `beamWidth` are known statically).
    */
  private[BeamSearchDecoder] case class MergeBatchBeamsConverter(
      batchSize: Output[Int],
      beamWidth: Int
  ) extends OutputStructure.Converter {
    @throws[InvalidArgumentException]
    @throws[InvalidShapeException]
    override def apply[T](value: Output[T], shape: Option[Shape]): Output[T] = {
      implicit val evTTF: TF[T] = TF.fromDataType(value.dataType)
      val valueShape = tf.shape(value)
      val reshapedValue = tf.reshape(value, tf.concatenate(Seq(
        batchSize.expandDims(0) * Output[Int](beamWidth),
        valueShape(2 ::)), axis = 0))
      val staticBatchSize = Output.constantValue(batchSize).map(_.scalar).getOrElse(-1)
      val batchSizeBeamWidth = if (staticBatchSize != -1) staticBatchSize * beamWidth else -1
      val expectedReshapedShape = Shape(batchSizeBeamWidth) ++ shape.get
      if (!reshapedValue.shape.isCompatibleWith(expectedReshapedShape)) {
        throw InvalidShapeException(
          "Unexpected behavior when reshaping between beam width and batch size. " +
              s"The reshaped tensor has shape: ${reshapedValue.shape}. " +
              s"We expected it to have shape [batchSize, beamWidth, depth] == $expectedReshapedShape. " +
              "Perhaps you forgot to create a zero state with batchSize = encoderBatchSize * beamWidth?")
      }
      reshapedValue.setShape(expectedReshapedShape)
      reshapedValue
    }

    @throws[InvalidArgumentException]
    override def apply[T](value: tf.data.Dataset[T], shape: Option[Shape]): tf.data.Dataset[T] = {
      throw InvalidArgumentException("Unsupported argument type for use with the beam search decoder.")
    }
  }

  /** Gathers the right indices from the provided `gatherFrom` value. This works by reshaping all tensors in
    * `gatherFrom` to `gatherShape` (e.g., `Seq(-1)`) and then gathering from that according to the `gatherIndices`,
    * which are offset by the right amount in order to preserve the batch order.
    *
    * @param  gatherIndices Indices that we use to gather from `gatherFrom`.
    * @param  gatherFrom    Value to gather from.
    * @param  batchSize     Input batch size.
    * @param  rangeSize     Number of values in each range. Likely equal to the beam width.
    * @param  gatherShape   What we should reshape `gatherFrom` to in order to preserve the correct values. An example
    *                       is when `gatherFrom` is the attention from an `AttentionWrapperState` with shape
    *                       `[batchSize, beamWidth, attentionSize]`. There, we want to preserve the `attentionSize`
    *                       elements, and so `gatherShape` is set to `Seq(batchSize * beamWidth, -1)`. Then, upon
    *                       reshape, we still have the `attentionSize` elements, as desired.
    * @return Value containing the gathered tensors of shapes `tf.shape(gatherFrom)(0 :: 1 + gatherShape.size())`.
    * @throws InvalidArgumentException If `gatherFrom` is of an unsupported type.
    */
  @throws[InvalidArgumentException]
  private[BeamSearchDecoder] def gather[T](
      gatherIndices: Output[Int],
      gatherFrom: Output[T],
      batchSize: Output[Int],
      rangeSize: Output[Int],
      gatherShape: Seq[Output[Int]],
      name: String = "GatherTensorHelper"
  ): Output[T] = {
    implicit val evTTF: TF[T] = TF.fromDataType(gatherFrom.dataType)
    tf.nameScope(name) {
      val range = (tf.range(0, batchSize) * rangeSize).expandDims(1)
      val reshapedGatherIndices = (gatherIndices + range).reshape(Shape(-1))
      val output = tf.gather(gatherFrom.reshape(tf.stack(gatherShape)), reshapedGatherIndices, axis = 0)
      val finalShape = tf.shape(gatherFrom).slice(0 :: (1 + gatherShape.size))
      tf.reshape(output, finalShape, name = "Output")
    }
  }

  /** Checks are known statically and can be reshaped to `[batchSize, beamSize, -1]` and logs a warning
    * if they cannot. */
  private[BeamSearchDecoder] def checkStaticBatchBeam(
      shape: Shape,
      batchSize: Int,
      beamWidth: Int
  ): Boolean = {
    if (batchSize != -1 && shape(0) != -1 &&
        (shape(0) != batchSize * beamWidth ||
            (shape.rank > 1 && shape(1) != -1 &&
                (shape(0) != batchSize || shape(1) != beamWidth)))) {
      val reshapedShape = Shape(batchSize, beamWidth, -1)
      logger.warn(
        s"Tensor array reordering expects elements to be reshapable to '$reshapedShape' which is incompatible with " +
            s"the current shape '$shape'. Consider setting `reorderTensorArrays` to `false` to disable tensor array " +
            "reordering during the beam search.")
      false
    } else {
      true
    }
  }

  /** Returns an assertion op checking that the elements of the stacked tensor array (i.e., `tensor`) can be reshaped
    * to `[batchSize, beamSize, -1]`. At this point, the tensor array elements have a known rank of at least `1`. */
  private[BeamSearchDecoder] def checkBatchBeam[T](
      tensor: Output[T],
      batchSize: Output[Int],
      beamWidth: Output[Int]
  ): UntypedOp = {
    implicit val evTF: TF[T] = TF.fromDataType(tensor.dataType)
    tf.assert(
      condition = {
        val shape = tf.shape(tensor).castTo[Int]
        if (tensor.rank == 2)
          tf.equal(shape(1), batchSize * beamWidth)
        else
          tf.logicalOr(
            tf.equal(shape(1), batchSize * beamWidth),
            tf.logicalAnd(
              tf.equal(shape(1), batchSize),
              tf.equal(shape(2), beamWidth)))
      },
      data = Seq(Tensor(
        "Tensor array reordering expects elements to be reshapable to '[batchSize, beamSize, -1]' which is " +
            s"incompatible with the dynamic shape of '${tensor.name}' elements. Consider setting " +
            "`reorderTensorArrays` to `false` to disable tensor array reordering during the beam search."
      ).toOutput))
  }

  /** Maybe sorts the search states within a tensor array. The input tensor array corresponds to the symbol to be
    * sorted. This will only be sorted if it is a tensor array of size `maxTime` that contains tensors with shape
    * `[batchSize, beamWidth, s]` or `[batchSize * beamWidth, s]`, where `s` is the depth shape.
    *
    * The converter returns a tensor array where the search states are sorted in each tensor, or `value` itself, if it
    * is not a tensor array or does not meet the shape requirements.
    *
    * @param  sequenceLengths Tensor containing the sequence lengths, with shape `[batchSize, beamWidth]`.
    * @param  parentIDs       Tensor containing the parent indices, with shape `[maxTime, batchSize, beamWidth]`.
    * @param  batchSize       Batch size.
    * @param  beamWidth       Beam width.
    */
  case class MaybeSortTensorArrayBeamsConverter(
      sequenceLengths: Output[Int],
      parentIDs: Output[Int],
      batchSize: Output[Int],
      beamWidth: Int
  ) extends OutputStructure.Converter {
    override def apply[T](value: TensorArray[T], shape: Option[Shape]): TensorArray[T] = {
      if ((!value.inferShape || value.elementShape.isEmpty) ||
          value.elementShape.get(0) == -1 ||
          value.elementShape.get(1) < 1
      ) {
        implicit val evTTF: TF[T] = TF.fromDataType(value.dataType)
        val shape = value.elementShape match {
          case Some(s) if value.inferShape => Shape(s(0))
          case _ => Shape(-1)
        }
        logger.warn(
          s"The tensor array '${value.handle.name}' in the cell state is not amenable to sorting based on the beam " +
              s"search result. For a tensor array to be sorted, its elements shape must be defined and have at least " +
              s"a rank of 1. However, the elements shape in the provided tensor array is: $shape.")
        value
      } else if (!checkStaticBatchBeam(
        shape = Shape(value.elementShape.get(0)),
        batchSize = Output.constantValue(batchSize).map(_.scalar).getOrElse(-1),
        beamWidth = beamWidth)
      ) {
        value
      } else {
        implicit val evTTF: TF[T] = TF.fromDataType(value.dataType)
        val stackedTensorArray = value.stack()
        tf.createWith(controlDependencies = Set(checkBatchBeam(stackedTensorArray, batchSize, beamWidth))) {
          val maxTime = tf.shape(parentIDs).castTo[Int].slice(0)
          val batchSize = tf.shape(parentIDs).castTo[Int].slice(1)
          val beamWidth = tf.shape(parentIDs).castTo[Int].slice(2)

          // Generate search state indices that will be reordered by the `gatherTree` op.
          val searchStateIndices = tf.tile(
            tf.range(0, beamWidth).slice(NewAxis, NewAxis),
            tf.stack[Int](Seq(maxTime, batchSize, 1)))
          val mask = tf.sequenceMask(sequenceLengths, maxTime).castTo[Int].transpose(Seq(2, 0, 1))

          // Use `beamWidth + 1` to mark the end of the beam.
          val maskedSearchStateIndices = (searchStateIndices * mask) + (1 - mask) * (beamWidth + 1)
          val maxSequenceLengths = sequenceLengths.max(Tensor[Int](1)).castTo[Int]
          var sortedSearchStateIndices = gatherTree(
            maskedSearchStateIndices, parentIDs, maxSequenceLengths, beamWidth + 1)

          // For out of range steps, we simply copy the same beam.
          sortedSearchStateIndices = tf.select(mask.toBoolean, sortedSearchStateIndices, searchStateIndices)

          // Generate indices for `gatherND`.
          val timeIndices = tf.tile(
            tf.range(0, maxTime).slice(NewAxis, NewAxis),
            tf.stack[Int](Seq(1, batchSize, beamWidth)))
          val batchIndices = tf.tile(
            tf.range(0, batchSize).slice(NewAxis, NewAxis),
            tf.stack[Int](Seq(1, maxTime, beamWidth))).transpose(Seq(1, 0, 2))
          val indices = tf.stack(Seq(timeIndices, batchIndices, sortedSearchStateIndices), -1)

          // Gather from a tensor with collapsed additional dimensions.
          val finalShape = tf.shape(stackedTensorArray)
          val gatherFrom = stackedTensorArray.reshape(tf.stack[Int](Seq(maxTime, batchSize, beamWidth, -1)))

          // TODO: [TYPES] !!! This cast is wrong. I need to figure out the right behavior for this.
          tf.gatherND(gatherFrom, indices).reshape(finalShape).asInstanceOf[TensorArray[T]]
        }
      }
    }

    @throws[InvalidArgumentException]
    override def apply[T](value: tf.data.Dataset[T], shape: Option[Shape]): tf.data.Dataset[T] = {
      throw InvalidArgumentException("Unsupported argument type for use with the beam search decoder.")
    }
  }
}
