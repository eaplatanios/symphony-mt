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

package org.platanios.symphony.mt.translators

import org.platanios.symphony.mt.core.{Configuration, Language, Translator}
import org.platanios.symphony.mt.data.Datasets
import org.platanios.symphony.mt.data.Datasets.MTTextLinesDataset
import org.platanios.symphony.mt.translators.PairwiseTranslator._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.StopCriteria
import org.platanios.tensorflow.api.learn.hooks.StepHookTrigger

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class PairwiseTranslator(
    override val configuration: Configuration = Configuration()
) extends Translator(configuration) {
  private[this] val estimators: mutable.Map[(Int, Int), MTPairwiseEstimator] = mutable.Map.empty

  // Create the input and the train input parts of the model.
  private[this] val seqShape   = Shape(configuration.batchSize, -1)
  private[this] val lenShape   = Shape(configuration.batchSize)
  private[this] val input      = tf.learn.Input((INT32, INT32), (seqShape, lenShape))
  private[this] val trainInput = tf.learn.Input((INT32, INT32, INT32), (seqShape, seqShape, lenShape))

  override def train(datasets: Seq[Translator.DatasetPair], stopCriteria: StopCriteria): Unit = {
    datasets
        .groupBy(p => (p.sourceLanguage.id, p.targetLanguage.id))
        .foreach {
          case (languageIdPair, datasetPairs) =>
            val srcLang = datasetPairs.head.sourceLanguage
            val tgtLang = datasetPairs.head.targetLanguage
            val workingDir = configuration.workingDir.resolve(s"${srcLang.abbreviation}-${tgtLang.abbreviation}")
            val srcVocab = srcLang.vocabulary()
            val tgtVocab = tgtLang.vocabulary()
            val srcVocabSize = srcLang.vocabularySize
            val tgtVocabSize = tgtLang.vocabularySize
            val srcDataset = Datasets.joinDatasets(datasetPairs.map(_.sourceDataset))
            val tgtDataset = Datasets.joinDatasets(datasetPairs.map(_.targetDataset))
            val trainDataset = Datasets.createTrainDataset(
              srcDataset, tgtDataset, srcVocab, tgtVocab, configuration.batchSize,
              configuration.beginOfSequenceToken, configuration.endOfSequenceToken,
              configuration.sourceReverse, configuration.randomSeed, configuration.numBuckets,
              configuration.sourceMaxLength, configuration.targetMaxLength, configuration.parallelIterations,
              configuration.dataBufferSize, configuration.dataDropCount, configuration.dataNumShards,
              configuration.dataShardIndex)
            val estimator = estimators.getOrElse(languageIdPair, {
              val tLayer = trainLayer(srcVocabSize, tgtVocabSize, srcVocab, tgtVocab)
              val iLayer = inferLayer(srcVocabSize, tgtVocabSize, srcVocab, tgtVocab)
              val model = tf.learn.Model(
                input = input,
                layer = iLayer,
                trainLayer = tLayer,
                trainInput = trainInput,
                loss = lossLayer(),
                optimizer = optimizer())
              val summariesDir = workingDir.resolve("summaries")
              val tensorBoardConfig = {
                if (configuration.launchTensorBoard)
                  tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 1)
                else
                  null
              }
              tf.learn.InMemoryEstimator(
                model,
                tf.learn.Configuration(Some(workingDir)),
                stopCriteria,
                Set(
                  tf.learn.StepRateHook(log = false, summaryDirectory = summariesDir, trigger = StepHookTrigger(100)),
                  tf.learn.SummarySaverHook(summariesDir, StepHookTrigger(10)),
                  tf.learn.CheckpointSaverHook(workingDir, StepHookTrigger(1000))),
                tensorBoardConfig = tensorBoardConfig)
            })
            estimator.train(trainDataset, stopCriteria)
        }
  }

  override def translate(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataset: MTTextLinesDataset
  ): MTTextLinesDataset = ???

  protected def trainLayer(
      srcVocabSize: Int,
      tgtVocabSize: Int,
      srcVocab: tf.LookupTable,
      tgtVocab: tf.LookupTable
  ): MTTrainLayer

  protected def inferLayer(
      srcVocabSize: Int,
      tgtVocabSize: Int,
      srcVocab: tf.LookupTable,
      tgtVocab: tf.LookupTable
  ): MTInferLayer

  protected def lossLayer(): MTLossLayer

  protected def optimizer(): tf.train.Optimizer
}

object PairwiseTranslator {
  type MTTrainLayer = tf.learn.Layer[((Output, Output), (Output, Output, Output)), (Output, Output)]
  type MTInferLayer = tf.learn.Layer[(Output, Output), (Output, Output)]
  type MTLossLayer = tf.learn.Layer[((Output, Output), (Output, Output, Output)), Output]

  type MTPairwiseEstimator = tf.learn.Estimator[
      (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape), (Output, Output),
      ((Tensor, Tensor), (Tensor, Tensor, Tensor)), ((Output, Output), (Output, Output, Output)),
      ((DataType, DataType), (DataType, DataType, DataType)), ((Shape, Shape), (Shape, Shape, Shape)),
      ((Output, Output), (Output, Output, Output))]
}
