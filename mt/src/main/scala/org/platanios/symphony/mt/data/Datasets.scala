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

package org.platanios.symphony.mt.data

import org.platanios.tensorflow.api._

/** Contains utilities for dealing with machine translation datasets.
  *
  * @author Emmanouil Antonios Platanios
  */
object Datasets {
  type MTTextLinesDataset = tf.data.Dataset[Tensor, Output, DataType, Shape]
  type MTInferDataset = tf.data.Dataset[(Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape)]

  type MTTrainDataset = tf.data.Dataset[
      ((Tensor, Tensor), (Tensor, Tensor, Tensor)),
      ((Output, Output), (Output, Output, Output)),
      ((DataType, DataType), (DataType, DataType, DataType)),
      ((Shape, Shape), (Shape, Shape, Shape))]

  def joinDatasets(datasets: Seq[MTTextLinesDataset]): MTTextLinesDataset = {
    datasets.reduce((d1, d2) => d1.concatenate(d2))
  }

  def createInferDataset(
      srcDataset: MTTextLinesDataset,
      srcVocabularyTable: tf.LookupTable,
      dataConfig: DataConfig,
      batchSize: Int
  ): MTInferDataset = {
    val srcEosId = srcVocabularyTable.lookup(tf.constant(dataConfig.endOfSequenceToken)).cast(INT32)

    val batchingFn = (dataset: MTInferDataset) => {
      dataset.dynamicPaddedBatch(
        batchSize,
        // The first entry represents the source line rows, which are unknown-length vectors.
        // The last entry is the source row size, which is a scalar.
        (Shape(-1), Shape.scalar()),
        // We pad the source sequences with 'endSequenceToken' tokens. Though notice that we do
        // not generally need to do this since later on we will be masking out calculations past
        // the true sequence.
        (srcEosId, tf.zeros(INT32, Shape.scalar())))
    }

    val datasetBeforeBatching = srcDataset
        .map(o => tf.stringSplit(o.expandDims(0)).values)
        // Crop based on the maximum allowed sequence length.
        .transform(d => if (dataConfig.srcMaxLength != -1) d.map(dd => dd(0 :: dataConfig.srcMaxLength)) else d)
        // Reverse the source sequence if necessary.
        .transform(d => if (dataConfig.srcReverse) d.map(dd => tf.reverse(dd, axes = 0)) else d)
        // Convert the word strings to IDs. Word strings that are not in the vocabulary
        // get the lookup table's default value.
        .map(d => tf.cast(srcVocabularyTable.lookup(d), INT32))
        // Add sequence lengths.
        .map(d => (d, tf.size(d, INT32)))

    batchingFn(datasetBeforeBatching)
  }

  def createTrainDataset(
      srcDataset: MTTextLinesDataset,
      tgtDataset: MTTextLinesDataset,
      srcVocabularyTable: tf.LookupTable,
      tgtVocabularyTable: tf.LookupTable,
      dataConfig: DataConfig,
      batchSize: Int,
      repeat: Boolean = true,
      randomSeed: Option[Int] = None
  ): MTTrainDataset = {
    val actualBufferSize = if (dataConfig.bufferSize == -1L) 1000 * batchSize else dataConfig.bufferSize
    val srcEosId = srcVocabularyTable.lookup(tf.constant(dataConfig.endOfSequenceToken)).cast(INT32)
    val tgtBosId = tgtVocabularyTable.lookup(tf.constant(dataConfig.beginOfSequenceToken)).cast(INT32)
    val tgtEosId = tgtVocabularyTable.lookup(tf.constant(dataConfig.endOfSequenceToken)).cast(INT32)

    val batchingFn = (dataset: MTTrainDataset) => {
      dataset.dynamicPaddedBatch(
        batchSize,
        // The first three entries are the source and target line rows, which are unknown-length vectors.
        // The last two entries are the source and target row sizes, which are scalars.
        ((Shape(-1), Shape.scalar()), (Shape(-1), Shape(-1), Shape.scalar())),
        // We pad the source and target sequences with 'endSequenceToken' tokens. Though notice that we do not
        // generally need to do this since later on we will be masking out calculations past the true sequence.
        ((srcEosId, tf.zeros(INT32, Shape.scalar().toOutput())),
            (tgtEosId, tgtEosId, tf.zeros(INT32, Shape.scalar().toOutput()))))
    }

    val datasetBeforeBucketing =
      srcDataset.zip(tgtDataset)
          .shard(dataConfig.numShards, dataConfig.shardIndex)
          .drop(dataConfig.dropCount)
          .transform(d => {
            if (repeat)
              d.repeat().prefetch(actualBufferSize)
            else
              d
          })
          .shuffle(actualBufferSize, randomSeed)
          // Tokenize by splitting on white spaces.
          .map(
            d => (tf.stringSplit(d._1(NewAxis)).values, tf.stringSplit(d._2(NewAxis)).values),
            name = "Map/StringSplit")
          .prefetch(actualBufferSize)
          // Filter zero length input sequences and sequences exceeding the maximum length.
          .filter(d => tf.logicalAnd(tf.size(d._1) > 0, tf.size(d._2) > 0))
          // Crop based on the maximum allowed sequence lengths.
          .transform(d => {
            if (dataConfig.srcMaxLength != -1 && dataConfig.tgtMaxLength != -1)
              d.map(
                dd => (dd._1(0 :: dataConfig.srcMaxLength), dd._2(0 :: dataConfig.tgtMaxLength)),
                dataConfig.numParallelCalls, name = "Map/MaxLength").prefetch(actualBufferSize)
            else if (dataConfig.srcMaxLength != -1)
              d.map(
                dd => (dd._1(0 :: dataConfig.srcMaxLength), dd._2),
                dataConfig.numParallelCalls, name = "Map/MaxLength").prefetch(actualBufferSize)
            else if (dataConfig.tgtMaxLength != -1)
              d.map(
                dd => (dd._1, dd._2(0 :: dataConfig.tgtMaxLength)),
                dataConfig.numParallelCalls, name = "Map/MaxLength").prefetch(actualBufferSize)
            else
              d
          })
          // Reverse the source sequence if necessary.
          .transform(d => {
            if (dataConfig.srcReverse)
              d.map(dd => (tf.reverse(dd._1, axes = 0), dd._2), name = "Map/SrcReverse").prefetch(actualBufferSize)
            else
              d
          })
          // Convert the word strings to IDs. Word strings that are not in the vocabulary
          // get the lookup table's default value.
          .map(
            d => (
              tf.cast(srcVocabularyTable.lookup(d._1), INT32),
              tf.cast(tgtVocabularyTable.lookup(d._2), INT32)),
            dataConfig.numParallelCalls, name = "Map/VocabularyLookup")
          .prefetch(actualBufferSize)
          // Create a target input prefixed with 'beginSequenceToken'
          // and a target output suffixed with 'endSequenceToken'.
          .map(
            d => (
              d._1,
              tf.concatenate(Seq(tgtBosId.expandDims(0), d._2), axis = 0),
              tf.concatenate(Seq(d._2, tgtEosId.expandDims(0)), axis = 0)),
            dataConfig.numParallelCalls, name = "Map/AddDecoderOutput")
          .prefetch(actualBufferSize)
          // Add sequence lengths.
          .map(
            d => ((d._1, tf.size(d._1, INT32)), (d._2, d._3, tf.size(d._2, INT32))), dataConfig.numParallelCalls,
            name = "Map/AddLengths")
          .prefetch(actualBufferSize)

    if (dataConfig.numBuckets == 1) {
      batchingFn(datasetBeforeBucketing)
    } else {
      // Calculate the bucket width by using the maximum source sequence length, if provided. Pairs with length
      // [0, bucketWidth) go to bucket 0, length [bucketWidth, 2 * bucketWidth) go to bucket 1, etc. Pairs with length
      // over ((numBuckets - 1) * bucketWidth) all go into the last bucket.
      val bucketWidth = {
        if (dataConfig.srcMaxLength != -1)
          (dataConfig.srcMaxLength + dataConfig.numBuckets - 1) / dataConfig.numBuckets
        else
          10
      }

      def keyFn(element: ((Output, Output), (Output, Output, Output))): Output = {
        // Bucket sequence  pairs based on the length of their source sequence and target sequence.
        val bucketId = tf.maximum(
          tf.truncateDivide(element._1._2, bucketWidth),
          tf.truncateDivide(element._2._3, bucketWidth))
        tf.minimum(dataConfig.numBuckets, bucketId).cast(INT64)
      }

      def reduceFn(pair: (Output, MTTrainDataset)): MTTrainDataset = {
        batchingFn(pair._2)
      }

      datasetBeforeBucketing.groupByWindow(keyFn, reduceFn, _ => batchSize)
    }
  }
}
