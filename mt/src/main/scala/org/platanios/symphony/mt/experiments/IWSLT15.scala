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
import org.platanios.symphony.mt.Language.{english, vietnamese}
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.data.loaders.IWSLT15DatasetLoader
import org.platanios.symphony.mt.evaluation.{BLEU, BilingualEvaluator}
import org.platanios.symphony.mt.models.{Model, StateBasedModel}
import org.platanios.symphony.mt.models.rnn._
import org.platanios.symphony.mt.models.rnn.attention.LuongRNNAttention
import org.platanios.symphony.mt.translators.PairwiseTranslator
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api.learn.StopCriteria
import org.platanios.tensorflow.api.ops.training.optimizers.GradientDescent

import java.nio.file.{Path, Paths}

/**
  * @author Emmanouil Antonios Platanios
  */
object IWSLT15 extends App {
  val workingDir: Path = Paths.get("temp").resolve("pairwise")

  val srcLang: Language = english
  val tgtLang: Language = vietnamese

  val dataConfig = DataConfig(
    workingDir = Paths.get("temp").resolve("data"),
    numBuckets = 5,
    srcMaxLength = 50,
    tgtMaxLength = 50)

  val dataset: FileParallelDataset = IWSLT15DatasetLoader(srcLang, tgtLang, dataConfig).load()

  val env = Environment(
    workingDir = workingDir.resolve(s"${srcLang.abbreviation}-${tgtLang.abbreviation}"),
    numGPUs = 1,
    parallelIterations = 32,
    swapMemory = true,
    randomSeed = Some(10))

  val optConfig = Model.OptConfig(
    maxGradNorm = 5.0f,
    optimizer = GradientDescent(_, _, learningRateSummaryTag = "LearningRate"),
    learningRateInitial = 1.0f,
    learningRateDecayRate = 0.5f,
    learningRateDecaySteps = 12000 * 1 / (3 * 4),
    learningRateDecayStartStep = 12000 * 2 / 3,
    colocateGradientsWithOps = true)

  val logConfig = Model.LogConfig(
    logLossSteps = 100,
    logTrainEvalSteps = -1)

  def model(srcLang: Language, srcVocab: Vocabulary, tgtLang: Language, tgtVocab: Vocabulary, env: Environment) = {
    StateBasedModel(
      name = "Model",
      srcLanguage = srcLang, srcVocabulary = srcVocab,
      tgtLanguage = tgtLang, tgtVocabulary = tgtVocab,
      dataConfig = dataConfig,
      config = StateBasedModel.Config(
        env,
        UnidirectionalRNNEncoder(
          cell = BasicLSTM(forgetBias = 1.0f),
          numUnits = 512,
          numLayers = 2,
          residual = false,
          dropout = Some(0.2f),
          timeMajor = true),
        UnidirectionalRNNDecoder(
          cell = BasicLSTM(forgetBias = 1.0f),
          numUnits = 512,
          numLayers = 2,
          residual = false,
          dropout = Some(0.2f),
          attention = Some(LuongRNNAttention(scaled = true)),
          outputAttention = true,
          timeMajor = true,
          beamWidth = 10),
        timeMajor = true),
      optConfig = optConfig,
      logConfig = logConfig,
      // TODO: !!! Find a way to set the number of buckets to 1.
      trainEvalDataset = () => dataset.filterTypes(Train).toTFBilingual(srcLang, tgtLang, repeat = false, isEval = true),
      devEvalDataset = () => dataset.filterTypes(Dev).toTFBilingual(srcLang, tgtLang, repeat = false, isEval = true),
      testEvalDataset = () => dataset.filterTypes(Test).toTFBilingual(srcLang, tgtLang, repeat = false, isEval = true))
  }

  val translator = PairwiseTranslator(env, model)
  translator.train(dataset, StopCriteria.steps(12000))

  val evaluator = BilingualEvaluator(Seq(BLEU()), srcLang, tgtLang, dataset.filterTypes(Test))
  println(evaluator.evaluate(translator).values.head.scalar.asInstanceOf[Float])
}
