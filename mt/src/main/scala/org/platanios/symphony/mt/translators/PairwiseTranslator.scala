///* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
// *
// * Licensed under the Apache License, Version 2.0 (the "License"); you may not
// * use this file except in compliance with the License. You may obtain a copy of
// * the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// * License for the specific language governing permissions and limitations under
// * the License.
// */
//
//package org.platanios.symphony.mt.translators
//
//import org.platanios.symphony.mt.Language
//import org.platanios.symphony.mt.data.Datasets
//import org.platanios.symphony.mt.data.Datasets.{MTTextLinesDataset, MTTrainDataset}
//import org.platanios.symphony.mt.metrics.{BLEUTensorFlow, Perplexity}
//import org.platanios.symphony.mt.models.Model
//import org.platanios.symphony.mt.translators.PairwiseTranslator._
//import org.platanios.tensorflow.api._
//import org.platanios.tensorflow.api.learn.{Mode, StopCriteria}
//import org.platanios.tensorflow.api.learn.hooks.StepHookTrigger
//import org.platanios.tensorflow.api.ops.training.optimizers.decay.ExponentialDecay
//
//import scala.collection.mutable
//
///**
//  * @author Emmanouil Antonios Platanios
//  */
//abstract class PairwiseTranslator(
//    override val model: Model,
//) extends Translator(model) {
//  private[this] val estimators: mutable.Map[(Language, Language), MTPairwiseEstimator] = mutable.Map.empty
//
//  override def train(
//      trainDatasets: Seq[Translator.DatasetPair],
//      devDatasets: Seq[Translator.DatasetPair] = null,
//      testDatasets: Seq[Translator.DatasetPair] = null,
//      stopCriteria: StopCriteria = StopCriteria(Some(model.trainConfig.numSteps))
//  ): Unit = {
//    val groupedTrainDatasets = groupAndJoinDatasets(trainDatasets)
//    val groupedDevDatasets = groupAndJoinDatasets(devDatasets)
//    val groupedTestDatasets = groupAndJoinDatasets(testDatasets)
//    groupedTrainDatasets.keys.foreach(pair => {
//      val srcLang = pair._1
//      val tgtLang = pair._2
//      val workingDir = configuration.workingDir.resolve(s"${srcLang.abbreviation}-${tgtLang.abbreviation}")
//      val srcVocab = srcLang.vocabulary
//      val tgtVocab = tgtLang.vocabulary
//      val srcVocabSize = srcLang.vocabularySize
//      val tgtVocabSize = tgtLang.vocabularySize
//      val srcTrainDataset = groupedTrainDatasets(pair)._1
//      val tgtTrainDataset = groupedTrainDatasets(pair)._2
//      val srcDevDataset = groupedDevDatasets(pair)._1
//      val tgtDevDataset = groupedDevDatasets(pair)._2
//      val srcTestDataset = groupedTestDatasets(pair)._1
//      val tgtTestDataset = groupedTestDatasets(pair)._2
//      val estimator = estimators.getOrElse((pair._1, pair._2), {
//        val tLayer = trainLayer(srcVocabSize, tgtVocabSize, srcVocab, tgtVocab)
//        val iLayer = inferLayer(srcVocabSize, tgtVocabSize, srcVocab, tgtVocab)
//        val model = tf.learn.Model(
//          input = input,
//          layer = iLayer,
//          trainLayer = tLayer,
//          trainInput = trainInput,
//          loss = lossLayer(),
//          optimizer = optimizer(),
//          clipGradients = tf.learn.ClipGradientsByGlobalNorm(configuration.trainMaxGradNorm),
//          colocateGradientsWithOps = configuration.trainColocateGradientsWithOps)
//        val summariesDir = workingDir.resolve("summaries")
//        val tensorBoardConfig = {
//          if (configuration.launchTensorBoard)
//            tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 1)
//          else
//            null
//        }
//        var hooks = Set[tf.learn.Hook](
//          tf.learn.StepRateLogger(log = false, summaryDir = summariesDir, trigger = StepHookTrigger(100)),
//          tf.learn.SummarySaver(summariesDir, StepHookTrigger(configuration.trainSummarySteps)),
//          tf.learn.CheckpointSaver(workingDir, StepHookTrigger(configuration.trainCheckpointSteps)))
//        if (configuration.logLossSteps > 0)
//          hooks += PerplexityLogger(log = true, trigger = StepHookTrigger(configuration.logLossSteps))
//        if (configuration.logTrainEvalSteps > 0)
//          hooks += tf.learn.Evaluator(
//            log = true, summariesDir,
//            () => createTrainDataset(
//              srcTrainDataset, tgtTrainDataset, srcVocab, tgtVocab, configuration.logEvalBatchSize, repeat = false, 1),
//            Seq(BLEUTensorFlow()), StepHookTrigger(configuration.logTrainEvalSteps),
//            triggerAtEnd = true, name = "Train Evaluation")
//        if (configuration.logDevEvalSteps > 0 && srcDevDataset != null && tgtDevDataset != null)
//          hooks += tf.learn.Evaluator(
//            log = true, summariesDir,
//            () => createTrainDataset(
//              srcDevDataset, tgtDevDataset, srcVocab, tgtVocab, configuration.logEvalBatchSize, repeat = false, 1),
//            Seq(BLEUTensorFlow()), StepHookTrigger(configuration.logDevEvalSteps),
//            triggerAtEnd = true, name = "Dev Evaluation")
//        if (configuration.logTestEvalSteps > 0 && srcTestDataset != null && tgtTestDataset != null)
//          hooks += tf.learn.Evaluator(
//            log = true, summariesDir,
//            () => createTrainDataset(
//              srcTestDataset, tgtTestDataset, srcVocab, tgtVocab, configuration.logEvalBatchSize, repeat = false, 1),
//            Seq(BLEUTensorFlow()), StepHookTrigger(configuration.logTestEvalSteps),
//            triggerAtEnd = true, name = "Test Evaluation")
//        tf.learn.InMemoryEstimator(
//          model, tf.learn.Configuration(Some(workingDir), randomSeed = configuration.randomSeed),
//          stopCriteria, hooks, tensorBoardConfig = tensorBoardConfig)
//      })
//      estimator.train(trainDataset, stopCriteria)
//    })
//  }
//
//  private[this] def groupAndJoinDatasets(
//      datasets: Seq[Translator.DatasetPair]
//  ): Map[(Language, Language), (MTTextLinesDataset, MTTextLinesDataset)] = {
//    if (datasets == null) {
//      Map.empty.withDefault(_ => (null, null))
//    } else {
//      datasets
//          .groupBy(p => (p.srcLanguage, p.tgtLanguage)).values
//          .map(p => (
//              (p.head.srcLanguage, p.head.tgtLanguage),
//              (Datasets.joinDatasets(p.map(_.srcDataset)), Datasets.joinDatasets(p.map(_.tgtDataset)))))
//          .toMap
//    }
//  }
//
//  private[this] def createTrainDataset(
//      srcDataset: MTTextLinesDataset,
//      tgtDataset: MTTextLinesDataset,
//      srcVocab: () => tf.LookupTable,
//      tgtVocab: () => tf.LookupTable,
//      batchSize: Int,
//      repeat: Boolean,
//      numBuckets: Int
//  ): MTTrainDataset = {
//    Datasets.createTrainDataset(
//      srcDataset, tgtDataset, srcVocab(), tgtVocab(), batchSize,
//      configuration.beginOfSequenceToken, configuration.endOfSequenceToken,
//      repeat, configuration.dataSrcReverse, configuration.randomSeed, numBuckets,
//      configuration.dataSrcMaxLength, configuration.dataTgtMaxLength, configuration.dataNumParallelCalls,
//      configuration.dataBufferSize, configuration.dataDropCount, configuration.dataNumShards,
//      configuration.dataShardIndex)
//  }
//
//  override def translate(
//      srcLanguage: Language,
//      tgtLanguage: Language,
//      dataset: MTTextLinesDataset
//  ): MTTextLinesDataset = ???
//
//  protected def trainLayer(
//      srcVocabSize: Int,
//      tgtVocabSize: Int,
//      srcVocab: () => tf.LookupTable,
//      tgtVocab: () => tf.LookupTable
//  ): MTTrainLayer
//
//  protected def inferLayer(
//      srcVocabSize: Int,
//      tgtVocabSize: Int,
//      srcVocab: () => tf.LookupTable,
//      tgtVocab: () => tf.LookupTable
//  ): MTInferLayer
//
//  protected def lossLayer(): MTLossLayer = {
//    new tf.learn.Layer[((Output, Output), (Output, Output, Output)), Output]("PairwiseTranslationLoss") {
//      override val layerType: String = "PairwiseTranslationLoss"
//
//      override def forward(
//          input: ((Output, Output), (Output, Output, Output)),
//          mode: Mode
//      ): LayerInstance[((Output, Output), (Output, Output, Output)), Output] = {
//        val loss = tf.sum(tf.sequenceLoss(
//          input._1._1, input._2._2,
//          weights = tf.sequenceMask(input._1._2, tf.shape(input._1._1)(1), dataType = input._1._1.dataType),
//          averageAcrossTimeSteps = false, averageAcrossBatch = true))
//        tf.summary.scalar("Loss", loss)
//        LayerInstance(input, loss)
//      }
//    }
//  }
//
//  protected def optimizer(): tf.train.Optimizer = {
//    val decay = ExponentialDecay(
//      configuration.trainLearningRateDecayRate,
//      configuration.trainLearningRateDecaySteps,
//      staircase = true,
//      configuration.trainLearningRateDecayStartStep)
//    configuration.trainOptimizer(configuration.trainLearningRateInitial, decay)
//  }
//
//  /** Returns the maximum sequence length to consider while decoding during inference, given the provided source
//    * sequence length. */
//  protected def inferMaxLength(srcLength: Output): Output = {
//    if (configuration.dataTgtMaxLength != -1)
//      tf.constant(configuration.dataTgtMaxLength)
//    else
//      tf.round(tf.max(srcLength) * configuration.modelDecodingMaxLengthFactor).cast(INT32)
//  }
//}
//
//object PairwiseTranslator {
//  type MTTrainLayer = tf.learn.Layer[((Output, Output), (Output, Output, Output)), (Output, Output)]
//  type MTInferLayer = tf.learn.Layer[(Output, Output), (Output, Output)]
//  type MTLossLayer = tf.learn.Layer[((Output, Output), (Output, Output, Output)), Output]
//
//  type MTPairwiseEstimator = tf.learn.Estimator[
//      (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape), (Output, Output),
//      ((Tensor, Tensor), (Tensor, Tensor, Tensor)), ((Output, Output), (Output, Output, Output)),
//      ((DataType, DataType), (DataType, DataType, DataType)), ((Shape, Shape), (Shape, Shape, Shape)),
//      ((Output, Output), (Output, Output, Output))]
//}
