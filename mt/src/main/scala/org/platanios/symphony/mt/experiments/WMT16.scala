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

package org.platanios.symphony.mt.experiments

import org.platanios.symphony.mt.{Environment, Language, LogConfig}
import org.platanios.symphony.mt.Language.{english, german}
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.data.loaders.WMT16DatasetLoader
import org.platanios.symphony.mt.models.attention.BahdanauAttention
import org.platanios.symphony.mt.models.rnn._
import org.platanios.symphony.mt.models.{Model, StateBasedModel}
import org.platanios.symphony.mt.translators.PairwiseTranslator
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api.learn.StopCriteria
import org.platanios.tensorflow.api.ops.training.optimizers.GradientDescent

import java.nio.file.{Path, Paths}

/**
  * @author Emmanouil Antonios Platanios
  */
object WMT16 extends App {
  val workingDir: Path = Paths.get("temp")

  val srcLang: Language = german
  val tgtLang: Language = english

  val dataConfig = DataConfig(
    workingDir = workingDir.resolve("data"),
    loaderTokenize = true,
    loaderSentenceLengthBounds = Some((1, 80)),
    numBuckets = 5,
    srcMaxLength = 50,
    tgtMaxLength = 50)

  val dataset: FileParallelDataset = WMT16DatasetLoader(srcLang, tgtLang, dataConfig).load()

  val env = Environment(
    workingDir = workingDir.resolve(s"${srcLang.abbreviation}-${tgtLang.abbreviation}"),
    numGPUs = 4,
    parallelIterations = 32,
    swapMemory = true,
    randomSeed = Some(10))

  val logConfig = LogConfig(
    logLossSteps = 100,
    logTrainEvalSteps = -1)

  def model(
      srcLang: Language,
      srcVocab: Vocabulary,
      tgtLang: Language,
      tgtVocab: Vocabulary,
      env: Environment
  ): Model = {
    StateBasedModel(
      name = "Model",
      srcLang = srcLang, srcVocab = srcVocab,
      tgtLang = tgtLang, tgtVocab = tgtVocab,
      StateBasedModel.Config(
        env,
        GNMTEncoder(
          cell = BasicLSTM(forgetBias = 1.0f),
          numUnits = 1024,
          numBiLayers = 1,
          numUniLayers = 3,
          numUniResLayers = 2,
          dropout = Some(0.2f),
          timeMajor = true),
        GNMTDecoder(
          cell = BasicLSTM(forgetBias = 1.0f),
          numUnits = 1024,
          numLayers = 1 + 3, // Number of encoder bidirectional and unidirectional layers
          numResLayers = 2,
          attention = BahdanauAttention(normalized = true),
          dropout = Some(0.2f),
          useNewAttention = true,
          timeMajor = true,
          beamWidth = 10,
          lengthPenaltyWeight = 1.0f),
        timeMajor = true,
        maxGradNorm = 5.0f,
        optimizer = GradientDescent(_, _, learningRateSummaryTag = "LearningRate"),
        learningRateInitial = 1.0f,
        learningRateDecayRate = 0.5f,
        learningRateDecaySteps = 340000 * 1 / (2 * 10),
        learningRateDecayStartStep = 340000 * 2,
        colocateGradientsWithOps = true),
      // TODO: !!! Find a way to set the number of buckets to 1.
      trainEvalDataset = () => dataset.filterTypes(Train).toTFBilingual(srcLang, tgtLang, repeat = false, isEval = true),
      devEvalDataset = () => dataset.filterTypes(Dev).toTFBilingual(srcLang, tgtLang, repeat = false, isEval = true),
      testEvalDataset = () => dataset.filterTypes(Test).toTFBilingual(srcLang, tgtLang, repeat = false, isEval = true),
      dataConfig, logConfig)
  }

  val translator = PairwiseTranslator(env, model)
  translator.train(dataset, StopCriteria.steps(340000))
}
