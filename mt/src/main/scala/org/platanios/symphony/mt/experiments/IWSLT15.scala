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

import org.platanios.symphony.mt.{Environment, Language, LogConfig}
import org.platanios.symphony.mt.data.Datasets.MTTextLinesDataset
import org.platanios.symphony.mt.data.{DataConfig, Vocabulary}
import org.platanios.symphony.mt.data.loaders.IWSLT15Loader
import org.platanios.symphony.mt.models.{InferConfig, Model, TrainConfig}
import org.platanios.symphony.mt.models.attention.{BahdanauAttention, LuongAttention}
import org.platanios.symphony.mt.models.rnn._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.StopCriteria
import org.platanios.tensorflow.api.ops.io.data.TextLinesDataset
import org.platanios.tensorflow.api.ops.training.optimizers.GradientDescent

import java.nio.file.{Path, Paths}

/**
  * @author Emmanouil Antonios Platanios
  */
object IWSLT15 extends App {
  val workingDir: Path = Paths.get("temp")
  val dataDir   : Path = workingDir.resolve("data").resolve("iwslt15.en-vi")

  IWSLT15Loader.download(workingDir.resolve("data"), IWSLT15Loader.EnglishVietnamese)

  // Create the languages and their corresponding vocabularies
  val srcLang : Language   = Language("English", "en")
  val tgtLang : Language   = Language("Vietnamese", "vi")
  val srcVocab: Vocabulary = Vocabulary(dataDir.resolve("vocab.en"))
  val tgtVocab: Vocabulary = Vocabulary(dataDir.resolve("vocab.vi"))

  // Create the datasets
  val srcTrainDataset: MTTextLinesDataset = TextLinesDataset(dataDir.resolve("train.en").toAbsolutePath.toString)
  val tgtTrainDataset: MTTextLinesDataset = TextLinesDataset(dataDir.resolve("train.vi").toAbsolutePath.toString)
  val srcDevDataset  : MTTextLinesDataset = TextLinesDataset(dataDir.resolve("tst2012.en").toAbsolutePath.toString)
  val tgtDevDataset  : MTTextLinesDataset = TextLinesDataset(dataDir.resolve("tst2012.vi").toAbsolutePath.toString)
  val srcTestDataset : MTTextLinesDataset = TextLinesDataset(dataDir.resolve("tst2013.en").toAbsolutePath.toString)
  val tgtTestDataset : MTTextLinesDataset = TextLinesDataset(dataDir.resolve("tst2013.vi").toAbsolutePath.toString)

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
  //  val config = Model.Config(
  //    UnidirectionalRNNEncoder(
  //      srcLang, srcVocab, env,
  //      cell = BasicLSTM(forgetBias = 1.0f),
  //      numUnits = 512,
  //      numLayers = 2,
  //      residual = false,
  //      dropout = Some(0.2f),
  //      timeMajor = true),
  //    UnidirectionalRNNDecoder(
  //      tgtLang, tgtVocab, env, dataConfig, inferConfig,
  //      cell = BasicLSTM(forgetBias = 1.0f),
  //      numUnits = 512,
  //      numLayers = 2,
  //      residual = false,
  //      dropout = Some(0.2f),
  //      attention = Some(LuongAttention(scaled = true)),
  //      outputAttention = true,
  //      timeMajor = true),
  //    timeMajor = true)

  val gnmtConfig = GNMTConfig(
    srcLang, tgtLang, srcVocab, tgtVocab, env, dataConfig, inferConfig,
    cell = BasicLSTM(forgetBias = 1.0f),
    numUnits = 512,
    numBiLayers = 1,
    numUniLayers = 2,
    numUniResLayers = 1,
    dropout = Some(0.2f),
    attention = BahdanauAttention(normalized = true),
    useNewAttention = false,
    timeMajor = true)

  val model = Model(
    gnmtConfig, srcLang, tgtLang, srcVocab, tgtVocab,
    srcTrainDataset, tgtTrainDataset, srcDevDataset, tgtDevDataset, srcTestDataset, tgtTestDataset,
    env, dataConfig, trainConfig, inferConfig, logConfig, "Model")

  model.train(StopCriteria(Some(trainConfig.numSteps)))
}
