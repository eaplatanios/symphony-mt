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
import org.platanios.symphony.mt.Language.{English, Vietnamese}
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.data.loaders.IWSLT15DatasetLoader
import org.platanios.symphony.mt.data.processors.{MosesCleaner, MosesTokenizer}
import org.platanios.symphony.mt.models.{Model, ParameterManager, RNNModel}
import org.platanios.symphony.mt.models.rnn._
import org.platanios.symphony.mt.models.rnn.attention.LuongRNNAttention
import org.platanios.symphony.mt.vocabulary.{SimpleVocabularyGenerator, Vocabulary}
import org.platanios.tensorflow.api._

import java.nio.file.{Path, Paths}

/**
  * @author Emmanouil Antonios Platanios
  */
object IWSLT15 extends App {
  val workingDir: Path = Paths.get("temp").resolve("iwslt15-rnn")

  val languagePairs: Set[(Language, Language)] = Set((English, Vietnamese))

  val dataConfig = DataConfig(
    workingDir = Paths.get("temp").resolve("data"),
    loaderTokenizer = MosesTokenizer(),
    loaderCleaner = MosesCleaner(),
    loaderVocab = GeneratedVocabulary(SimpleVocabularyGenerator(sizeThreshold = 50000, countThreshold = 5)),
    numBuckets = 10,
    srcMaxLength = 100,
    tgtMaxLength = 100)

  val (datasets, languages): (Seq[FileParallelDataset], Seq[(Language, Vocabulary)]) = {
    loadDatasets(languagePairs.toSeq.map(l => IWSLT15DatasetLoader(l._1, l._2, dataConfig)), Some(workingDir))
  }

  val env = Environment(
    workingDir = workingDir.resolve(languages.map(_._1.abbreviation).mkString("-")),
    allowSoftPlacement = true,
    logDevicePlacement = false,
    gpuAllowMemoryGrowth = false,
    useXLA = false,
    numGPUs = 1,
    parallelIterations = 32,
    swapMemory = true,
    randomSeed = Some(10))

  val optConfig = Model.OptConfig(
    maxGradNorm = 100.0f,
    optimizer = tf.train.AMSGrad(learningRateSummaryTag = "LearningRate"))
    // optimizer = tf.train.GradientDescent(
    //   1.0f, tf.train.ExponentialDecay(decayRate = 0.5f, decaySteps = 12000 * 1 / (3 * 4), startStep = 12000 * 2 / 3),
    //   learningRateSummaryTag = "LearningRate"))

  val logConfig = Model.LogConfig(
    logLossSteps = 100,
    launchTensorBoard = true)

  val model = RNNModel(
    name = "Model",
    languages = languages,
    dataConfig = dataConfig,
    config = RNNModel.Config(
      env,
      ParameterManager(
        wordEmbeddingsSize = 512,
        variableInitializer = tf.VarianceScalingInitializer(
          1.0f,
          tf.VarianceScalingInitializer.FanAverageScalingMode,
          tf.VarianceScalingInitializer.UniformDistribution)),
      BidirectionalRNNEncoder(
        cell = BasicLSTM(forgetBias = 1.0f),
        numUnits = 512,
        numLayers = 2,
        residual = false,
        dropout = Some(0.2f)),
      UnidirectionalRNNDecoder(
        cell = BasicLSTM(forgetBias = 1.0f),
        numUnits = 512,
        numLayers = 2,
        residual = false,
        dropout = Some(0.2f),
        attention = Some(LuongRNNAttention(scaled = true)),
        outputAttention = true),
      labelSmoothing = 0.1f,
      timeMajor = true,
      beamWidth = 10),
    optConfig = optConfig,
    logConfig = logConfig,
    // TODO: !!! Find a way to set the number of buckets to 1.
    evalDatasets = datasets.flatMap(d => Seq(
      // ("IWSLT15/dev2010", d.filterTags(IWSLT15DatasetLoader.Dev2010)),
      // ("IWSLT15/tst2010", d.filterTags(IWSLT15DatasetLoader.Test2010)),
      // ("IWSLT15/tst2011", d.filterTags(IWSLT15DatasetLoader.Test2011)),
      // ("IWSLT15/tst2012", d.filterTags(IWSLT15DatasetLoader.Test2012)),
      ("IWSLT15/tst2013", d.filterTags(IWSLT15DatasetLoader.Test2013)))
    ))

  model.train(datasets.map(_.filterTypes(Train)), tf.learn.StopCriteria.steps(12000))

  // val evaluator = BilingualEvaluator(Seq(BLEU()), srcLanguage, tgtLanguage, dataset.filterTypes(Test))
  // println(evaluator.evaluate(model).values.head.scalar.asInstanceOf[Float])
}
