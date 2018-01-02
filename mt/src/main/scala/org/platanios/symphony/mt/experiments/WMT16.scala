/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.symphony.mt.Language.{English, German}
import org.platanios.symphony.mt.data.Datasets.MTTextLinesDataset
import org.platanios.symphony.mt.data.managers.WMT16Manager
import org.platanios.symphony.mt.data.{DataConfig, ParallelDataset, Vocabulary}
import org.platanios.symphony.mt.models.attention.BahdanauAttention
import org.platanios.symphony.mt.models.rnn._
import org.platanios.symphony.mt.models.{InferConfig, StateBasedModel, TrainConfig}
import org.platanios.symphony.mt.{Environment, Language, LogConfig}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.StopCriteria
import org.platanios.tensorflow.api.ops.io.data.TextLinesDataset
import org.platanios.tensorflow.api.ops.training.optimizers.GradientDescent

import java.nio.file.{Path, Paths}

/**
  * @author Emmanouil Antonios Platanios
  */
object WMT16 extends App {
  val workingDir: Path = Paths.get("temp")

  // Create the languages.
  val srcLang: Language = German
  val tgtLang: Language = English

  val parallelDataset: ParallelDataset = WMT16Manager(srcLang, tgtLang).download(workingDir.resolve("data"))

  // Create the vocabularies and the datasets
  val srcVocab       : Vocabulary         = Vocabulary(parallelDataset.vocabulary()(srcLang))
  val tgtVocab       : Vocabulary         = Vocabulary(parallelDataset.vocabulary()(tgtLang))
  val srcTrainDataset: MTTextLinesDataset = TextLinesDataset(parallelDataset.trainCorpus()(srcLang).toAbsolutePath.toString)
  val tgtTrainDataset: MTTextLinesDataset = TextLinesDataset(parallelDataset.trainCorpus()(tgtLang).toAbsolutePath.toString)
  val srcDevDataset  : MTTextLinesDataset = TextLinesDataset(parallelDataset.devCorpus()(srcLang).toAbsolutePath.toString)
  val tgtDevDataset  : MTTextLinesDataset = TextLinesDataset(parallelDataset.devCorpus()(tgtLang).toAbsolutePath.toString)
  val srcTestDataset : MTTextLinesDataset = TextLinesDataset(parallelDataset.testCorpus()(srcLang).toAbsolutePath.toString)
  val tgtTestDataset : MTTextLinesDataset = TextLinesDataset(parallelDataset.testCorpus()(tgtLang).toAbsolutePath.toString)

  // Create general configuration settings
  val env = Environment(
    workingDir = Paths.get("temp").resolve(s"${srcLang.abbreviation}-${tgtLang.abbreviation}"),
    numGPUs = 4,
    parallelIterations = 32,
    swapMemory = true,
    randomSeed = Some(10))

  val dataConfig = DataConfig(
    numBuckets = 5,
    srcMaxLength = 50,
    tgtMaxLength = 50)

  val trainConfig = TrainConfig(
    batchSize = 128,
    maxGradNorm = 5.0f,
    numSteps = 12000,
    optimizer = GradientDescent(_, _, learningRateSummaryTag = "LearningRate"),
    learningRateInitial = 1.0f,
    learningRateDecayRate = 0.5f,
    learningRateDecaySteps = 12000 * 1 / (3 * 4),
    learningRateDecayStartStep = 12000 * 2 / 3,
    colocateGradientsWithOps = true)

  val inferConfig = InferConfig(
    batchSize = 32,
    beamWidth = 10)

  val logConfig = LogConfig(
    logLossSteps = 100,
    logTrainEvalSteps = -1)

  // Create a translator
  val config = GNMTConfig(
    srcLang, tgtLang, srcVocab, tgtVocab, env, dataConfig, inferConfig,
    cell = BasicLSTM(forgetBias = 1.0f),
    numUnits = 512,
    numBiLayers = 1,
    numUniLayers = 3,
    numUniResLayers = 2,
    dropout = Some(0.2f),
    attention = BahdanauAttention(normalized = true),
    useNewAttention = false,
    timeMajor = true)

  val model = StateBasedModel(
    config, srcLang, tgtLang, srcVocab, tgtVocab,
    srcTrainDataset, tgtTrainDataset, srcDevDataset, tgtDevDataset, srcTestDataset, tgtTestDataset,
    env, dataConfig, trainConfig, inferConfig, logConfig, "Model")

  model.train(srcTrainDataset, tgtTrainDataset, StopCriteria(Some(trainConfig.numSteps)))
}
