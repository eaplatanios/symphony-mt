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

import java.nio.file.{Path, Paths}

import org.platanios.symphony.mt.Language.{english, vietnamese}
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.data.loaders.IWSLT15DatasetLoader
import org.platanios.symphony.mt.models.rnn._
import org.platanios.symphony.mt.models.rnn.attention.LuongRNNAttention
import org.platanios.symphony.mt.models.{LanguageEmbeddingsPairParameterManager, Model, RNNModel}
import org.platanios.symphony.mt.{Environment, Language}
import org.platanios.tensorflow.api._

/**
  * @author Emmanouil Antonios Platanios
  */
object IWSLT15LanguageEmbeddings extends App {
  val workingDir: Path = Paths.get("temp").resolve("pairwise")

  val srcLanguage: Language = english
  val tgtLanguage: Language = vietnamese

  val dataConfig = DataConfig(
    workingDir = Paths.get("temp").resolve("data"),
    loaderVocab = MergedVocabularies,
    numBuckets = 5,
    srcMaxLength = 50,
    tgtMaxLength = 50)

  val dataset: FileParallelDataset = IWSLT15DatasetLoader(srcLanguage, tgtLanguage, dataConfig).load()

  val env = Environment(
    workingDir = workingDir.resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"),
    numGPUs = 1,
    parallelIterations = 32,
    swapMemory = true,
    randomSeed = Some(10))

  val optConfig = Model.OptConfig(
    maxGradNorm = 5.0f,
    optimizer = tf.train.GradientDescent(
      1.0f, tf.train.ExponentialDecay(decayRate = 0.5f, decaySteps = 12000 * 1 / (3 * 4), startStep = 12000 * 2 / 3),
      learningRateSummaryTag = "LearningRate"))

  val logConfig = Model.LogConfig(logLossSteps = 100)

  val model = RNNModel(
    name = "Model",
    languages = Seq(srcLanguage -> dataset.vocabulary(srcLanguage), tgtLanguage -> dataset.vocabulary(tgtLanguage)),
    dataConfig = dataConfig,
    config = RNNModel.Config(
      env,
      LanguageEmbeddingsPairParametersManager(
        languageEmbeddingsSize = 256,
        wordEmbeddingsSize = 256),
      BidirectionalRNNEncoder(
        cell = BasicLSTM(forgetBias = 1.0f),
        numUnits = 256,
        numLayers = 2,
        residual = false,
        dropout = Some(0.2f)),
      UnidirectionalRNNDecoder(
        cell = BasicLSTM(forgetBias = 1.0f),
        numUnits = 256,
        numLayers = 2,
        residual = false,
        dropout = Some(0.2f),
        attention = Some(LuongRNNAttention(scaled = true)),
        outputAttention = true),
      labelSmoothing = 0.0f,
      timeMajor = true,
      beamWidth = 10),
    optConfig = optConfig,
    logConfig = logConfig,
    // TODO: !!! Find a way to set the number of buckets to 1.
    evalDatasets = Seq(
      ("IWSLT15", dataset.filterTypes(Dev).filterLanguages(srcLanguage, tgtLanguage)),
      ("IWSLT15", dataset.filterTypes(Test).filterLanguages(srcLanguage, tgtLanguage))))

  model.train(dataset.filterTypes(Train), tf.learn.StopCriteria.steps(12000))

  // val evaluator = BilingualEvaluator(Seq(BLEU()), srcLanguage, tgtLanguage, dataset.filterTypes(Test))
  // println(evaluator.evaluate(model).values.head.scalar.asInstanceOf[Float])
}
