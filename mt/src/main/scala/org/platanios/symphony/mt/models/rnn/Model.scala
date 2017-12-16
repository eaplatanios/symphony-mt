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

package org.platanios.symphony.mt.models.rnn

import org.platanios.symphony.mt.data.Datasets
import org.platanios.symphony.mt.data.Datasets.{MTTextLinesDataset, MTTrainDataset}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.StopCriteria

/**
  * @author Emmanouil Antonios Platanios
  */
trait Model[S, SS] {
  val name         : String
  val configuration: Configuration[S, SS]

  protected def createTrainDataset(
      srcDataset: MTTextLinesDataset,
      tgtDataset: MTTextLinesDataset,
      srcVocab: () => tf.LookupTable,
      tgtVocab: () => tf.LookupTable,
      batchSize: Int,
      repeat: Boolean,
      numBuckets: Int
  ): MTTrainDataset = {
    Datasets.createTrainDataset(
      srcDataset, tgtDataset, srcVocab(), tgtVocab(), batchSize,
      configuration.beginOfSequenceToken, configuration.endOfSequenceToken,
      repeat, configuration.dataSrcReverse, configuration.randomSeed, numBuckets,
      configuration.dataSrcMaxLength, configuration.dataTgtMaxLength, configuration.dataNumParallelCalls,
      configuration.dataBufferSize, configuration.dataDropCount, configuration.dataNumShards,
      configuration.dataShardIndex)
  }

  def train(stopCriteria: StopCriteria = StopCriteria(Some(configuration.trainNumSteps))): Unit
}
