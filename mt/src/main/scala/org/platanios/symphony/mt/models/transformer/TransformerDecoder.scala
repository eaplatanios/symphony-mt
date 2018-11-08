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

package org.platanios.symphony.mt.models.transformer

import org.platanios.symphony.mt.models.{Context, Sequences}
import org.platanios.symphony.mt.models.Transformation.Decoder
import org.platanios.symphony.mt.models.helpers.decoders.{BasicDecoder, BeamSearchDecoder}
import org.platanios.symphony.mt.models.transformer.helpers.Attention.MultiHeadAttentionCache
import org.platanios.symphony.mt.models.transformer.helpers._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.tf.{RNNCell, RNNTuple}

/**
  * @author Emmanouil Antonios Platanios
  */
class TransformerDecoder[T: TF : IsHalfOrFloatOrDouble](
    val numLayers: Int,
    val attentionPrependMode: AttentionPrependMode = AttentionPrependInputsMaskedAttention,
    val useSelfAttentionProximityBias: Boolean = false,
    val positionEmbeddings: PositionEmbeddings = FixedSinusoidPositionEmbeddings(1.0f, 1e4f),
    val postPositionEmbeddingsDropout: Float = 0.1f,
    val usePadRemover: Boolean = true,
    val layerPreprocessors: Seq[LayerProcessor] = Seq(
      Normalize(LayerNormalization(), 1e-6f)),
    val layerPostprocessors: Seq[LayerProcessor] = Seq(
      Dropout(0.1f, broadcastAxes = Set(1)),
      AddResidualConnection),
    val attentionKeysDepth: Int = 128,
    val attentionValuesDepth: Int = 128,
    val attentionNumHeads: Int = 8,
    val selfAttention: Attention = DotProductAttention(0.1f, Set.empty, "DotProductAttention"),
    val feedForwardLayer: FeedForwardLayer = DenseReLUDenseFeedForwardLayer(256, 128, 0.1f, Set.empty, "FeedForward")
) extends Decoder[EncodedSequences[T]] {
  override def applyTrain(
      encodedSequences: EncodedSequences[T]
  )(implicit context: Context): Sequences[Float] = {
    // TODO: What if no target sequences are provided?
    val tgtSequences = context.tgtSequences.get

    // Shift the target sequence one step forward so the decoder learns to output the next word.
    val bosToken = tf.constant[String](context.dataConfig.beginOfSequenceToken)
    val tgtBosId = context.parameterManager.stringToIndexLookup(context.tgtLanguageID)(bosToken)
    val batchSize = tf.shape(tgtSequences.sequences).slice(0)
    val shiftedTgtSequences = tf.concatenate(Seq(
      tf.fill[Int, Int](tf.stack[Int](Seq(batchSize, 1)))(tgtBosId),
      tgtSequences.sequences), axis = 1)
    val shiftedTgtSequenceLengths = tgtSequences.lengths + 1
    val shiftedTgtSequencesMaxLength = tf.shape(shiftedTgtSequences).slice(1)

    // Embed the target sequences.
    val embeddedTgtSequences = embeddings(shiftedTgtSequences)

    // Perform some pre-processing to the token embeddings sequence.
    val padding = tf.sequenceMask(shiftedTgtSequenceLengths, shiftedTgtSequencesMaxLength, name = "Padding").toFloat
    var decoderSelfAttentionBias = attentionPrependMode(padding)
    if (useSelfAttentionProximityBias)
      decoderSelfAttentionBias += Attention.attentionBiasProximal(shiftedTgtSequencesMaxLength)
    var decoderInputSequences = positionEmbeddings.addTo(embeddedTgtSequences)
    if (context.mode.isTraining)
      decoderInputSequences = tf.dropout(decoderInputSequences, 1.0f - postPositionEmbeddingsDropout)

    val encodedSequencesMaxLength = tf.shape(encodedSequences.states).slice(1)
    val inputPadding = tf.sequenceMask(encodedSequences.lengths, encodedSequencesMaxLength, name = "Padding").toFloat
    val encoderDecoderAttentionBias = Attention.attentionBiasIgnorePadding(inputPadding)

    // Obtain the output projection layer weights.
    val wordEmbeddingsSize = context.parameterManager.wordEmbeddingsType.embeddingsSize
    val outputWeights = context.parameterManager.getProjectionToWords(
      wordEmbeddingsSize, context.tgtLanguageID).castTo[T]
    val outputLayer = this.outputLayer(outputWeights)(_)

    // Finally, apply the decoding model.
    val (decodedSequences, _) = decode(
      encoderOutput = encodedSequences.states.toFloat,
      decoderInputSequences = decoderInputSequences,
      decoderSelfAttentionBias = decoderSelfAttentionBias,
      decoderSelfAttentionCache = None,
      encoderDecoderAttentionBias = encoderDecoderAttentionBias,
      encoderDecoderAttentionCache = None)
    val outputSequences = outputLayer(decodedSequences)
    Sequences(outputSequences.toFloat, encodedSequences.lengths)
  }

  override def applyInfer(
      encodedSequences: EncodedSequences[T]
  )(implicit context: Context): Sequences[Int] = {
    val wordEmbeddingsSize = context.parameterManager.wordEmbeddingsType.embeddingsSize

    // Determine the maximum allowed sequence length to consider while decoding.
    val maxDecodingLength = {
      if (!context.mode.isTraining && context.dataConfig.tgtMaxLength != -1)
        tf.constant(context.dataConfig.tgtMaxLength)
      else
        tf.round(tf.max(encodedSequences.lengths).toFloat *
            context.modelConfig.inferenceConfig.maxDecodingLengthFactor).toInt
    }

    val positionEmbeddings = this.positionEmbeddings.get(maxDecodingLength + 1, wordEmbeddingsSize).castTo[T]
    var decoderSelfAttentionBias = Attention.attentionBiasLowerTriangular(maxDecodingLength)
    if (useSelfAttentionProximityBias)
      decoderSelfAttentionBias += Attention.attentionBiasProximal(maxDecodingLength)

    // Pre-compute the encoder-decoder attention keys and values.
    val encoderOutput = encodedSequences.states
    val encoderDecoderAttentionKeys = Attention.computeAttentionComponent(
      encoderOutput, attentionKeysDepth, name = "K").toFloat
    val encoderDecoderAttentionValues = Attention.computeAttentionComponent(
      encoderOutput, attentionValuesDepth, name = "V").toFloat
    val encoderDecoderAttentionCache = MultiHeadAttentionCache(
      keys = Attention.splitHeads(encoderDecoderAttentionKeys, attentionNumHeads),
      values = Attention.splitHeads(encoderDecoderAttentionValues, attentionNumHeads))

    // Create the decoder RNN cell.
    val zero = tf.constant[Int](0)
    val one = tf.constant[Int](1)
    val cell = new RNNCell[Output[T], (EncodedSequences[T], Output[T], Seq[MultiHeadAttentionCache[Float]]), Shape, ((Shape, Shape), Shape, Seq[(Shape, Shape)])] {
      override def outputShape: Shape = {
        val wordEmbeddingsSize = context.parameterManager.wordEmbeddingsType.embeddingsSize
        Shape(wordEmbeddingsSize)
      }

      override def stateShape: ((Shape, Shape), Shape, Seq[(Shape, Shape)]) = {
        ((Shape(-1, wordEmbeddingsSize), Shape()),
            Shape(-1, wordEmbeddingsSize),
            (0 until numLayers).map(_ =>
              (Shape(attentionNumHeads, -1, attentionKeysDepth / attentionNumHeads),
                  Shape(attentionNumHeads, -1, attentionValuesDepth / attentionNumHeads))))
      }

      override def forward(
          input: RNNTuple[Output[T], (EncodedSequences[T], Output[T], Seq[MultiHeadAttentionCache[Float]])]
      ): RNNTuple[Output[T], (EncodedSequences[T], Output[T], Seq[MultiHeadAttentionCache[Float]])] = {
        val encodedSequences = input.state._1
        val encodedSequencesMaxLength = tf.shape(encodedSequences.states).slice(1)
        val inputPadding = tf.sequenceMask(encodedSequences.lengths, encodedSequencesMaxLength, name = "Padding")
        val encoderDecoderAttentionBias = Attention.attentionBiasIgnorePadding(inputPadding.toFloat)

        val step = tf.shape(input.state._2).slice(1)
        val currentStepPositionEmbeddings = positionEmbeddings(0).gather(step).expandDims(0).expandDims(1)
        val currentStepInput = input.output.expandDims(1) + currentStepPositionEmbeddings
        val concatenatedOutput = tf.concatenate(Seq(input.state._2, currentStepInput), axis = 1)
        val selfAttentionBias = tf.slice(
          decoderSelfAttentionBias(0, 0, ::, ::).gather(step),
          Seq(zero),
          Seq(step + one))
        val (output, updatedMultiHeadAttentionCache) = decode(
          encoderOutput = input.state._1.states.toFloat,
          decoderInputSequences = currentStepInput,
          decoderSelfAttentionBias = selfAttentionBias,
          decoderSelfAttentionCache = Some(input.state._3),
          encoderDecoderAttentionBias = encoderDecoderAttentionBias,
          encoderDecoderAttentionCache = Some(encoderDecoderAttentionCache))
        val currentStepOutput = output(::, 0, ::)
        RNNTuple(currentStepOutput, (input.state._1, concatenatedOutput, updatedMultiHeadAttentionCache.map(_.get)))
      }
    }

    // Create some constants that will be used during decoding.
    val bosToken = tf.constant[String](context.dataConfig.beginOfSequenceToken)
    val eosToken = tf.constant[String](context.dataConfig.endOfSequenceToken)
    val tgtVocabLookupTable = context.parameterManager.stringToIndexLookup(context.tgtLanguageID)
    val tgtBosID = tgtVocabLookupTable(bosToken).toInt
    val tgtEosID = tgtVocabLookupTable(eosToken).toInt

    // Obtain the output projection layer weights.
    val outputWeights = context.parameterManager.getProjectionToWords(
      wordEmbeddingsSize, context.tgtLanguageID).castTo[T]

    // Initialize the cache and the decoder RNN state.
    val embeddings = (ids: Output[Int]) => this.embeddings(ids)
    val batchSize = tf.shape(encodedSequences.lengths).slice(0)
    val multiHeadAttentionCache = (0 until numLayers).map(_ => MultiHeadAttentionCache(
      keys = Attention.splitHeads(tf.zeros[Float](tf.stack[Int](Seq(batchSize, zero, attentionKeysDepth))), attentionNumHeads),
      values = Attention.splitHeads(tf.zeros[Float](tf.stack[Int](Seq(batchSize, zero, attentionValuesDepth))), attentionNumHeads)))
    val initialState = (
        encodedSequences,
        tf.zeros[T](tf.stack[Int](Seq(batchSize, zero, wordEmbeddingsSize))),
        multiHeadAttentionCache)

    // Create the decoder RNN.
    val output = {
      if (context.modelConfig.inferenceConfig.beamWidth > 1) {
        val decoder = BeamSearchDecoder(
          cell, initialState, embeddings, tf.fill[Int, Int](batchSize.expandDims(0))(tgtBosID),
          tgtEosID, context.modelConfig.inferenceConfig.beamWidth, context.modelConfig.inferenceConfig.lengthPenalty,
          outputLayer(outputWeights))
        val tuple = decoder.decode(
          outputTimeMajor = context.modelConfig.timeMajor,
          maximumIterations = maxDecodingLength,
          parallelIterations = context.env.parallelIterations,
          swapMemory = context.env.swapMemory)
        Sequences(tuple._1.predictedIDs(---, 0), tuple._3(---, 0).toInt)
      } else {
        val decHelper = BasicDecoder.GreedyEmbeddingHelper[T, (EncodedSequences[T], Output[T], Seq[MultiHeadAttentionCache[Float]])](
          embeddingFn = embeddings,
          beginTokens = tf.fill[Int, Int](batchSize.expandDims(0))(tgtBosID),
          endToken = tgtEosID)
        val decoder = BasicDecoder(cell, initialState, decHelper, outputLayer(outputWeights))
        val tuple = decoder.decode(
          outputTimeMajor = context.modelConfig.timeMajor,
          maximumIterations = maxDecodingLength,
          parallelIterations = context.env.parallelIterations,
          swapMemory = context.env.swapMemory)
        Sequences(tuple._1.sample, tuple._3)
      }
    }

    Sequences(output.sequences(---, 0 :: -1), output.lengths - 1)
  }

  protected def embeddings(
      ids: Output[Int]
  )(implicit context: Context): Output[T] = {
    val embeddingsTable = context.parameterManager.wordEmbeddings(context.tgtLanguageID)
    embeddingsTable(ids).castTo[T]
  }

  protected def outputLayer(outputWeights: Output[T])(logits: Output[T]): Output[T] = {
    val reshapedLogits = tf.reshape(logits, Shape(-1, logits.shape(-1)))
    val product = tf.matmul(reshapedLogits, outputWeights)
    if (logits.shape(1) == -1 || outputWeights.shape(1) == -1) {
      tf.reshape(
        product,
        tf.concatenate(Seq(
          tf.shape(logits).slice(0 :: -1),
          tf.shape(outputWeights).slice(1, NewAxis)), axis = 0))
    } else {
      tf.reshape(product, logits.shape(0 :: -1) + outputWeights.shape(1))
    }
  }

  protected def decode(
      encoderOutput: Output[Float],
      decoderInputSequences: Output[T],
      decoderSelfAttentionBias: Output[Float],
      decoderSelfAttentionCache: Option[Seq[MultiHeadAttentionCache[Float]]],
      encoderDecoderAttentionBias: Output[Float],
      encoderDecoderAttentionCache: Option[MultiHeadAttentionCache[Float]]
  )(implicit context: Context): (Output[T], Seq[Option[MultiHeadAttentionCache[Float]]]) = {
    val wordEmbeddingsSize = context.parameterManager.wordEmbeddingsType.embeddingsSize

    // Add the multi-head attention and the feed-forward layers.
    // TODO: What about the padding remover?
    var x = decoderInputSequences.toFloat
    var y = x
    val updatedMultiHeadAttentionCache = (0 until numLayers).map(layer => {
      tf.variableScope(s"Layer$layer") {
        val updatedMultiHeadAttentionCache = tf.variableScope("SelfAttention") {
          val queryAntecedent = LayerProcessor.layerPreprocess(x, layerPreprocessors)
          val (output, updatedMultiHeadAttentionCache) = Attention.multiHeadAttention(
            queryAntecedent = queryAntecedent,
            memoryAntecedent = None,
            bias = decoderSelfAttentionBias,
            totalKeysDepth = attentionKeysDepth,
            totalValuesDepth = attentionValuesDepth,
            outputsDepth = wordEmbeddingsSize,
            numHeads = attentionNumHeads,
            attention = selfAttention,
            cache = decoderSelfAttentionCache.map(_ (layer)),
            name = "MultiHeadAttention")
          y = output
          x = LayerProcessor.layerPostprocess(x, y, layerPostprocessors)
          updatedMultiHeadAttentionCache
        }
        tf.variableScope("EncoderDecoderAttention") {
          val queryAntecedent = LayerProcessor.layerPreprocess(x, layerPreprocessors)
          y = Attention.multiHeadAttention(
            queryAntecedent = queryAntecedent,
            memoryAntecedent = Some(encoderOutput),
            bias = encoderDecoderAttentionBias,
            totalKeysDepth = attentionKeysDepth,
            totalValuesDepth = attentionValuesDepth,
            outputsDepth = wordEmbeddingsSize,
            numHeads = attentionNumHeads,
            attention = selfAttention,
            cache = encoderDecoderAttentionCache,
            name = "MultiHeadAttention")._1
          x = LayerProcessor.layerPostprocess(x, y, layerPostprocessors)
        }
        tf.variableScope("FeedForward") {
          val xx = LayerProcessor.layerPreprocess(x, layerPreprocessors)
          y = feedForwardLayer(xx, None)
          x = LayerProcessor.layerPostprocess(x, y, layerPostprocessors)
        }
        updatedMultiHeadAttentionCache
      }
    })

    // If normalization is done during layer preprocessing, then it should also be done on the output, since the
    // output can grow very large, being the sum of a whole stack of unnormalized layer outputs.
    (LayerProcessor.layerPreprocess(x, layerPreprocessors).castTo[T], updatedMultiHeadAttentionCache)
  }
}
