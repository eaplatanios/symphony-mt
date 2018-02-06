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
import org.platanios.symphony.mt.Language.{English, Vietnamese}
import org.platanios.symphony.mt.data.{DataConfig, Dataset}
import org.platanios.symphony.mt.data.datasets.IWSLT15Dataset
import org.platanios.symphony.mt.models.{InferConfig, StateBasedModel, TrainConfig}
import org.platanios.symphony.mt.models.attention.LuongAttention
import org.platanios.symphony.mt.models.rnn._
import org.platanios.tensorflow.api.learn.StopCriteria
import org.platanios.tensorflow.api.ops.training.optimizers.GradientDescent

import java.nio.file.{Path, Paths}


/**
  * @author Emmanouil Antonios Platanios
  */
object IWSLT15 extends App {
  val workingDir: Path = Paths.get("temp")

  // Create the languages.
  val srcLang: Language = English
  val tgtLang: Language = Vietnamese

  val dataset: Dataset.GroupedFiles = IWSLT15Dataset(workingDir.resolve("data"), srcLang, tgtLang).groupedFiles.withNewVocab()

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
  val config = StateBasedModel.Config(
    UnidirectionalRNNEncoder(
      srcLang, dataset.srcVocab, env,
      cell = BasicLSTM(forgetBias = 1.0f),
      numUnits = 512,
      numLayers = 2,
      residual = false,
      dropout = Some(0.2f),
      timeMajor = true),
    UnidirectionalRNNDecoder(
      tgtLang, dataset.tgtVocab, env, dataConfig, inferConfig,
      cell = BasicLSTM(forgetBias = 1.0f),
      numUnits = 512,
      numLayers = 2,
      residual = false,
      dropout = Some(0.2f),
      attention = Some(LuongAttention(scaled = true)),
      outputAttention = true,
      timeMajor = true),
    timeMajor = true)

  val trainDataset     = () => dataset.createTrainDataset(Dataset.TRAIN, trainConfig.batchSize, dataConfig, repeat = true)
  val trainEvalDataset = () => dataset.createTrainDataset(Dataset.TRAIN, logConfig.logEvalBatchSize, dataConfig.copy(numBuckets = 1), repeat = false)
  val devEvalDataset   = () => dataset.createTrainDataset(Dataset.DEV, logConfig.logEvalBatchSize, dataConfig.copy(numBuckets = 1), repeat = false)
  val testEvalDataset  = () => dataset.createTrainDataset(Dataset.TEST, logConfig.logEvalBatchSize, dataConfig.copy(numBuckets = 1), repeat = false)

  val model = StateBasedModel(
    config, srcLang, tgtLang, dataset.srcVocab, dataset.tgtVocab, trainEvalDataset, devEvalDataset, testEvalDataset,
    env, dataConfig, trainConfig, inferConfig, logConfig, "Model")

  model.train(trainDataset, StopCriteria.steps(trainConfig.numSteps))
}
