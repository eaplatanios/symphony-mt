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

package org.platanios.symphony.mt.models

import org.platanios.symphony.mt.{Environment, Language}
import org.platanios.symphony.mt.data.{DataConfig, TFBilingualDataset}
import org.platanios.symphony.mt.models.attention._
import org.platanios.symphony.mt.models.helpers.{Common, DecodeHelper, PadRemover}
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode

/**
  * @author Emmanouil Antonios Platanios
  */
class Transformer protected (
    override val name: String,
    override val srcLanguage: Language,
    override val srcVocabulary: Vocabulary,
    override val tgtLanguage: Language,
    override val tgtVocabulary: Vocabulary,
    override val dataConfig: DataConfig,
    override val config: Transformer.Config,
    override val optConfig: Model.OptConfig,
    override val logConfig : Model.LogConfig  = Model.LogConfig(),
    override val trainEvalDataset: () => TFBilingualDataset = null,
    override val devEvalDataset: () => TFBilingualDataset = null,
    override val testEvalDataset: () => TFBilingualDataset = null
) extends Model[Transformer.State](
  name, srcLanguage, srcVocabulary, tgtLanguage, tgtVocabulary, dataConfig, config, optConfig, logConfig,
  trainEvalDataset, devEvalDataset, testEvalDataset
) {
  override protected def encoder(input: (Output, Output), mode: Mode): Transformer.State = {
    val (inputTokens, inputLengths) = input
    val inputMaxLength = tf.shape(inputTokens)(1)

    // Obtain token embeddings for the input sequence.
    val embeddings = parametersManager.get("Embeddings", FLOAT32, Shape(srcVocabulary.size, config.hiddenSize))
    val embeddedInputs = tf.embeddingLookup(embeddings, inputTokens)

    // Perform some pre-processing to the token embeddings sequence.
    val padding = tf.sequenceMask(inputLengths, inputMaxLength, dataType = FLOAT32, name = "Padding")
    val attentionBias = Attention.attentionBiasIgnorePadding(padding)
    var encoderSelfAttentionBias = attentionBias
    val encoderDecoderAttentionBias = attentionBias
    if (config.useSelfAttentionProximityBias)
      encoderSelfAttentionBias += Attention.attentionBiasProximal(inputMaxLength)
    var encoderInput = config.positionEmbeddings.addTo(embeddedInputs)
    if (mode.isTraining)
      encoderInput = tf.dropout(encoderInput, 1.0f - config.postPositionEmbeddingsDropout)

    // Add the multi-head attention and the feed-forward layers.
    val padRemover = if (config.usePadRemover && !Common.isOnTPU) Some(PadRemover(padding)) else None
    var x = encoderInput
    var y = x
    (0 until config.encoderNumLayers).foreach(layer => {
      tf.createWithVariableScope(s"Layer$layer") {
        tf.createWithVariableScope("SelfAttention") {
          val queryAntecedent = LayerProcessor.layerPreprocess(x, config.layerPreprocessors)(mode, parametersManager)
          y = Attention.multiHeadAttention(
            queryAntecedent = queryAntecedent,
            memoryAntecedent = queryAntecedent,
            bias = encoderSelfAttentionBias,
            totalKeysDepth = config.attentionKeysDepth,
            totalValuesDepth = config.attentionValuesDepth,
            outputsDepth = config.hiddenSize,
            numHeads = config.attentionNumHeads,
            attention = config.encoderSelfAttention,
            name = "MultiHeadAttention")(mode, parametersManager)
          x = LayerProcessor.layerPostprocess(x, y, config.layerPostprocessors)(mode, parametersManager)
        }
        tf.createWithVariableScope("FeedForward") {
          val xx = LayerProcessor.layerPreprocess(x, config.layerPreprocessors)(mode, parametersManager)
          y = config.encoderFeedForwardLayer(xx, padRemover)(mode, parametersManager)
          x = LayerProcessor.layerPostprocess(x, y, config.layerPostprocessors)(mode, parametersManager)
        }
      }
    })

    // If normalization is done during layer preprocessing, then it should also be done on the output, since the output
    // can grow very large, being the sum of a whole stack of unnormalized layer outputs.
    val encoderOutput = LayerProcessor.layerPreprocess(x, config.layerPreprocessors)(mode, parametersManager)

    Transformer.State(encoderOutput, encoderDecoderAttentionBias)
  }

  override protected def decoder(
      input: Option[(Output, Output)],
      state: Option[Transformer.State],
      mode: Mode
  ): (Output, Output) = {
    // TODO: !!! Inference time.

    val cache: Option[Seq[Attention.Cache]] = None // TODO: !!!

    // TODO: Handle this shift more efficiently.
    // Shift the target sequence one step forward so the decoder learns to output the next word.
    val tgtBosId = tgtVocabulary.lookupTable().lookup(tf.constant(dataConfig.beginOfSequenceToken)).cast(INT32)
    val inputTokens = tf.concatenate(Seq(
      tf.fill(INT32, tf.stack(Seq(tf.shape(input.get._1)(0), 1)))(tgtBosId),
      input.get._1), axis = 1)
    val inputLengths = input.get._2 + 1
    val inputMaxLength = tf.shape(inputTokens)(1)

    // Obtain token embeddings for the input sequence.
    val embeddings = parametersManager.get("Embeddings", FLOAT32, Shape(tgtVocabulary.size, config.hiddenSize))
    val embeddedInputs = tf.embeddingLookup(embeddings, inputTokens)

    // Perform some pre-processing to the token embeddings sequence.
    val padding = tf.sequenceMask(inputLengths, inputMaxLength, dataType = FLOAT32, name = "Padding")
    var decoderSelfAttentionBias = config.attentionPrependMode(padding)
    if (config.useSelfAttentionProximityBias)
      decoderSelfAttentionBias += Attention.attentionBiasProximal(inputMaxLength)
    var decoderInput = config.positionEmbeddings.addTo(embeddedInputs)
    if (mode.isTraining)
      decoderInput = tf.dropout(decoderInput, 1.0f - config.postPositionEmbeddingsDropout)

    // Add the multi-head attention and the feed-forward layers.
    // TODO: What about the padding remover?
    var x = decoderInput
    var y = x
    (0 until config.decoderNumLayers).foreach(layer => {
      tf.createWithVariableScope(s"Layer$layer") {
        tf.createWithVariableScope("SelfAttention") {
          val queryAntecedent = LayerProcessor.layerPreprocess(x, config.layerPreprocessors)(mode, parametersManager)
          y = Attention.multiHeadAttention(
            queryAntecedent = queryAntecedent,
            memoryAntecedent = queryAntecedent,
            bias = decoderSelfAttentionBias,
            totalKeysDepth = config.attentionKeysDepth,
            totalValuesDepth = config.attentionValuesDepth,
            outputsDepth = config.hiddenSize,
            numHeads = config.attentionNumHeads,
            attention = config.decoderSelfAttention,
            cache = cache.map(_ (layer)),
            name = "MultiHeadAttention")(mode, parametersManager)
          x = LayerProcessor.layerPostprocess(x, y, config.layerPostprocessors)(mode, parametersManager)
        }
        state.foreach(encOutput => {
          // TODO: Add caching.
          tf.createWithVariableScope("EncoderDecoderAttention") {
            val queryAntecedent = LayerProcessor.layerPreprocess(x, config.layerPreprocessors)(mode, parametersManager)
            y = Attention.multiHeadAttention(
              queryAntecedent = queryAntecedent,
              memoryAntecedent = encOutput.output,
              bias = encOutput.encoderDecoderAttentionBias,
              totalKeysDepth = config.attentionKeysDepth,
              totalValuesDepth = config.attentionValuesDepth,
              outputsDepth = config.hiddenSize,
              numHeads = config.attentionNumHeads,
              attention = config.decoderSelfAttention,
              name = "MultiHeadAttention")(mode, parametersManager)
            x = LayerProcessor.layerPostprocess(x, y, config.layerPostprocessors)(mode, parametersManager)
          }
        })
        tf.createWithVariableScope("FeedForward") {
          val xx = LayerProcessor.layerPreprocess(x, config.layerPreprocessors)(mode, parametersManager)
          y = config.encoderFeedForwardLayer(xx, None)(mode, parametersManager)
          x = LayerProcessor.layerPostprocess(x, y, config.layerPostprocessors)(mode, parametersManager)
        }
      }
    })

    // If normalization is done during layer preprocessing, then it should also be done on the output, since the output
    // can grow very large, being the sum of a whole stack of unnormalized layer outputs.
    var decoderOutput = LayerProcessor.layerPreprocess(x, config.layerPreprocessors)(mode, parametersManager)

    val w = parametersManager.get(
      "DecoderOutputProjectionWeights", decoderOutput.dataType, Shape(decoderOutput.shape(-1), tgtVocabulary.size))
    decoderOutput = tf.linear(decoderOutput, w)

    (decoderOutput, inputLengths)
  }
}

object Transformer {
  def apply(
      name: String = "Transformer",
      srcLanguage: Language,
      srcVocabulary: Vocabulary,
      tgtLanguage: Language,
      tgtVocabulary: Vocabulary,
      dataConfig: DataConfig,
      config: Config,
      optConfig: Model.OptConfig,
      logConfig: Model.LogConfig,
      trainEvalDataset: () => TFBilingualDataset = null,
      devEvalDataset: () => TFBilingualDataset = null,
      testEvalDataset: () => TFBilingualDataset = null
  ): Transformer = {
    new Transformer(
      name, srcLanguage, srcVocabulary, tgtLanguage, tgtVocabulary, dataConfig, config, optConfig, logConfig,
      trainEvalDataset, devEvalDataset, testEvalDataset)
  }

  case class Config(
      override val env: Environment,
      useSelfAttentionProximityBias: Boolean,
      positionEmbeddings: PositionalEmbeddings,
      postPositionEmbeddingsDropout: Float,
      layerPreprocessors: Seq[LayerProcessor],
      layerPostprocessors: Seq[LayerProcessor],
      hiddenSize: Int,
      encoderNumLayers: Int,
      encoderSelfAttention: Attention,
      encoderFeedForwardLayer: FeedForwardLayer,
      decoderNumLayers: Int,
      decoderSelfAttention: Attention,
      attentionKeysDepth: Int,
      attentionValuesDepth: Int,
      attentionNumHeads: Int,
      attentionDropoutRate: Float,
      attentionDropoutBroadcastAxes: Set[Int],
      attentionPrependMode: AttentionPrependMode,
      usePadRemover: Boolean = true,
      override val summarySteps: Int = 100,
      override val checkpointSteps: Int = 1000
  ) extends Model.Config(env, timeMajor = false, summarySteps = summarySteps, checkpointSteps = checkpointSteps)

  val defaultConfig: Config = Config(
    env = null,
    useSelfAttentionProximityBias = false,
    positionEmbeddings = FixedSinusoidPositionalEmbeddings(1.0f, 1e4f),
    postPositionEmbeddingsDropout = 0.1f,
    layerPreprocessors = Seq(
      Normalize(LayerNormalization(), 1e-6f)),
    layerPostprocessors = Seq(
      Dropout(0.9f, broadcastAxes = Set(1)),
      AddResidualConnection),
    hiddenSize = 512,
    encoderNumLayers = 6,
    encoderSelfAttention = DotProductAttention(0.1f, Set.empty, "DotProductAttention"),
    encoderFeedForwardLayer = DenseReLUDenseFeedForwardLayer(2048, 512, 0.0f, Set.empty, "FeedForward"),
    decoderNumLayers = 6,
    decoderSelfAttention = DotProductAttention(0.1f, Set.empty, "DotProductAttention"),
    attentionKeysDepth = 0,
    attentionValuesDepth = 0,
    attentionNumHeads = 8,
    attentionDropoutRate = 0.1f,
    attentionDropoutBroadcastAxes = Set.empty,
    attentionPrependMode = AttentionPrependInputsMaskedAttention,
    usePadRemover = true)

  case class State(output: Output, encoderDecoderAttentionBias: Output)

  class GreedyDecodeHelper protected (val config: Config) extends helpers.DecodeHelper[State, Seq[Attention.Cache]] {
    override def batchSize(encoderOutput: State): Output = {
      tf.shape(encoderOutput.output)(0)
    }

    override def decode(
        encoderOutput: State,
        decodingFn: (Output, Output, State, Seq[Attention.Cache]) => (Output, Seq[Attention.Cache]),
        decodingLength: Int,
        endOfSequenceID: Output
    ): DecodeHelper.Result = {
      val batchSize = this.batchSize(encoderOutput)
      val cache = (0 until config.decoderNumLayers).map(_ => Attention.Cache(
        tf.zeros(FLOAT32, tf.stack(Seq(batchSize, 0, config.attentionKeysDepth))),
        tf.zeros(FLOAT32, tf.stack(Seq(batchSize, 0, config.attentionValuesDepth)))))

      type LoopVariables = (Output, Output, Output, Output, (Output, Output), Seq[(Output, Output)])

      /** Performs one step of greedy decoding. */
      def bodyFn(loopVariables: LoopVariables): LoopVariables = {
        val (i, finished, currentIDs, decodedIDs, encoderOutput, cache) = loopVariables
        val (logits, nextCache) = decodingFn(
          i, currentIDs, State(encoderOutput._1, encoderOutput._2),
          cache.map(c => Attention.Cache(c._1, c._2)))
        // TODO: Add support for sampling with temperature.
        var nextIDs = tf.argmax(logits, axes = -1)
        val nextFinished = tf.logicalOr(finished, tf.equal(nextIDs, endOfSequenceID))
        nextIDs = tf.expandDims(nextIDs, axis = 1)
        val nextDecodedIDs = tf.concatenate(Seq(decodedIDs, nextIDs), axis = 1)
        (i + 1, nextFinished, nextIDs, nextDecodedIDs, encoderOutput, nextCache.map(c => (c.k, c.v)))
      }

      val decodedIds = tf.zeros(INT64, tf.stack(Seq(batchSize, 0)))
      val finished = tf.fill(BOOLEAN, batchSize.expandDims(0))(false)
      val ids = tf.zeros(INT64, tf.stack(Seq(batchSize, 1)))
      val (_, _, _, finalDecodedIds, _, _) = tf.whileLoop(
        (lv: LoopVariables) => tf.logicalAnd(lv._1 < decodingLength, tf.logicalNot(tf.all(lv._2))),
        (lv: LoopVariables) => bodyFn(lv),
        loopVariables = (
            tf.constant(0), finished, ids, decodedIds,
            (encoderOutput.output, encoderOutput.encoderDecoderAttentionBias),
            cache.map(c => (c.k, c.v))),
        shapeInvariants = Some(
          (Shape(), Shape(-1), Shape(-1, -1), Shape(-1, -1),
              (DecodeHelper.stateShapeInvariants(encoderOutput.output),
                  DecodeHelper.stateShapeInvariants(encoderOutput.encoderDecoderAttentionBias)),
              Seq.fill(cache.length)((
                  Shape(-1, -1, config.attentionKeysDepth),
                  Shape(-1, -1, config.attentionValuesDepth))))))

      DecodeHelper.Result(finalDecodedIds)
    }
  }
}
