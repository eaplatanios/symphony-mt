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

import org.platanios.symphony.mt.{Environment, Language}
import org.platanios.symphony.mt.Language.{english, german}
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.data.loaders.WMT16DatasetLoader
import org.platanios.symphony.mt.models.{Model, ParametersManager, RNNModel}
import org.platanios.symphony.mt.models.rnn._
import org.platanios.symphony.mt.models.rnn.attention.BahdanauRNNAttention

import org.platanios.tensorflow.api._

import java.nio.file.{Path, Paths}

/**
  * @author Emmanouil Antonios Platanios
  */
object WMT16 extends App {
  val workingDir: Path = Paths.get("temp")

  val srcLanguage: Language = german
  val tgtLanguage: Language = english

  val dataConfig = DataConfig(
    workingDir = workingDir.resolve("data"),
    loaderTokenize = true,
    loaderSentenceLengthBounds = Some((1, 80)),
    numBuckets = 5,
    srcMaxLength = 50,
    tgtMaxLength = 50)

  val dataset: FileParallelDataset = WMT16DatasetLoader(srcLanguage, tgtLanguage, dataConfig).load()

  val env = Environment(
    workingDir = workingDir.resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"),
    numGPUs = 4,
    parallelIterations = 32,
    swapMemory = true,
    randomSeed = Some(10))

  val optConfig = Model.OptConfig(
    maxGradNorm = 5.0f,
    optimizer = tf.train.GradientDescent(
      1.0f, tf.train.ExponentialDecay(decayRate = 0.5f, decaySteps = 340000 * 1 / (2 * 10), startStep = 340000 / 2),
      learningRateSummaryTag = "LearningRate"))

  val logConfig = Model.LogConfig(logLossSteps = 100)

  val model = RNNModel(
    name = "Model",
    languages = Seq(srcLanguage -> dataset.vocabulary(srcLanguage), tgtLanguage -> dataset.vocabulary(tgtLanguage)),
    dataConfig = dataConfig,
    config = RNNModel.Config(
      env,
      ParametersManager(
        wordEmbeddingsSize = 1024,
        tf.VarianceScalingInitializer(
          1.0f,
          tf.VarianceScalingInitializer.FanAverageScalingMode,
          tf.VarianceScalingInitializer.UniformDistribution)),
      GNMTEncoder(
        cell = BasicLSTM(forgetBias = 1.0f),
        numUnits = 1024,
        numBiLayers = 1,
        numUniLayers = 3,
        numUniResLayers = 2,
        dropout = Some(0.2f)),
      GNMTDecoder(
        cell = BasicLSTM(forgetBias = 1.0f),
        numUnits = 1024,
        numLayers = 1 + 3, // Number of encoder bidirectional and unidirectional layers
        numResLayers = 2,
        attention = BahdanauRNNAttention(normalized = true),
        dropout = Some(0.2f),
        useNewAttention = true),
      labelSmoothing = 0.0f,
      timeMajor = true,
      beamWidth = 10,
      lengthPenaltyWeight = 1.0f),
    optConfig = optConfig,
    logConfig = logConfig,
    // TODO: !!! Find a way to set the number of buckets to 1.
    evalDatasets = Seq(
      ("WMT16", dataset.filterTypes(Dev).filterLanguages(srcLanguage, tgtLanguage)),
      ("WMT16", dataset.filterTypes(Test).filterLanguages(srcLanguage, tgtLanguage))))

  model.train(dataset, tf.learn.StopCriteria.steps(340000))
}
