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
import org.platanios.symphony.mt.Language.{English, German}
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.data.datasets.WMT16Dataset
import org.platanios.symphony.mt.models.attention.BahdanauAttention
import org.platanios.symphony.mt.models.rnn._
import org.platanios.symphony.mt.models.{InferConfig, StateBasedModel, TrainConfig}
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

  val srcLang: Language = German
  val tgtLang: Language = English

  val dataConfig = DataConfig(
    workingDir = workingDir.resolve("data"),
    loaderTokenize = true,
    loaderSentenceLengthBounds = Some((1, 80)),
    numBuckets = 5,
    srcMaxLength = 50,
    tgtMaxLength = 50)

  val dataset     : LoadedDataset              = WMT16Dataset(srcLang, tgtLang, dataConfig).load()
  val datasetFiles: LoadedDataset.GroupedFiles = dataset.files(srcLang, tgtLang)

  val env = Environment(
    workingDir = Paths.get("temp").resolve(s"${srcLang.abbreviation}-${tgtLang.abbreviation}"),
    numGPUs = 4,
    parallelIterations = 32,
    swapMemory = true,
    randomSeed = Some(10))

  val trainConfig = TrainConfig(
    maxGradNorm = 5.0f,
    optimizer = GradientDescent(_, _, learningRateSummaryTag = "LearningRate"),
    learningRateInitial = 1.0f,
    learningRateDecayRate = 0.5f,
    learningRateDecaySteps = 340000 * 1 / (2 * 10),
    learningRateDecayStartStep = 340000 * 2,
    stopCriteria = StopCriteria(maxSteps = Some(340000)),
    colocateGradientsWithOps = true)

  val inferConfig = InferConfig(
    beamWidth = 10,
    lengthPenaltyWeight = 1.0f)

  val logConfig = LogConfig(
    logLossSteps = 100,
    logTrainEvalSteps = -1)

  // Create a translator
  val config = GNMTConfig(
    srcLang, tgtLang, datasetFiles.srcVocab, datasetFiles.tgtVocab, env, dataConfig, inferConfig,
    cell = BasicLSTM(forgetBias = 1.0f),
    numUnits = 1024,
    numBiLayers = 1,
    numUniLayers = 3,
    numUniResLayers = 2,
    dropout = Some(0.2f),
    attention = BahdanauAttention(normalized = true),
    useNewAttention = false,
    timeMajor = true)

  val trainDataset     = () => datasetFiles.createTrainDataset(TRAIN_DATASET, repeat = true)
  val trainEvalDataset = () => datasetFiles.createTrainDataset(TRAIN_DATASET, repeat = false, dataConfig.copy(numBuckets = 1), isEval = true)
  val devEvalDataset   = () => datasetFiles.createTrainDataset(DEV_DATASET, repeat = false, dataConfig.copy(numBuckets = 1), isEval = true)
  val testEvalDataset  = () => datasetFiles.createTrainDataset(TEST_DATASET, repeat = false, dataConfig.copy(numBuckets = 1), isEval = true)

  val model = (srcLang: Language, tgtLang: Language, srcVocab: Vocabulary, tgtVocab: Vocabulary) => StateBasedModel(
    config, srcLang, tgtLang, srcVocab, tgtVocab,
    trainEvalDataset, devEvalDataset, testEvalDataset,
    env, dataConfig, trainConfig, inferConfig, logConfig, "Model")

  val translator = PairwiseTranslator(model)

  translator.train(dataset, trainReverse = false)
}
