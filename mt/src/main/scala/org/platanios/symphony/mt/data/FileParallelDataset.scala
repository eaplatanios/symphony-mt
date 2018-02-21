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

package org.platanios.symphony.mt.data

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._

import better.files.File

/**
  * @author Emmanouil Antonios Platanios
  */
class FileParallelDataset protected (
    override val name: String,
    override val vocabulary: Map[Language, Vocabulary],
    val dataConfig: DataConfig,
    val files: Map[Language, Seq[File]],
    val fileTypes: Seq[DatasetType] = null,
    val fileKeys: Seq[String] = null
) extends ParallelDataset {
  override def filterLanguages(languages: Language*): ParallelDataset = {
    languages.foreach(checkSupportsLanguage)
    FileParallelDataset(
      s"$name/${languages.map(_.abbreviation).mkString("-")}",
      vocabulary.filterKeys(languages.contains), dataConfig,
      files.filterKeys(languages.contains), fileTypes, fileKeys)
  }

  override def filterTypes(types: DatasetType*): ParallelDataset = {
    val filteredGroupedFiles = files.mapValues(_.zip(fileTypes).filter(f => types.contains(f._2)).map(_._1))
    val filteredFileTypes = fileTypes.filter(types.contains)
    val filteredFileKeys = fileKeys.zip(fileTypes).filter(f => types.contains(f._2)).map(_._1)
    FileParallelDataset(
      s"$name/${types.mkString("-")}", vocabulary, dataConfig,
      filteredGroupedFiles, filteredFileTypes, filteredFileKeys)
  }

  override def filterKeys(keys: String*): ParallelDataset = {
    require(fileKeys.nonEmpty, "Cannot filter a parallel dataset by file key when it contains no file keys.")
    val filteredGroupedFiles = files.mapValues(_.zip(fileKeys).filter(f => keys.contains(f._2)).map(_._1))
    val filteredFileTypes = fileKeys.zip(fileTypes).filter(f => keys.contains(f._1)).map(_._2)
    val filteredFileKeys = fileKeys.filter(keys.contains)
    FileParallelDataset(
      s"$name/${keys.mkString("-")}", vocabulary, dataConfig,
      filteredGroupedFiles, filteredFileTypes, filteredFileKeys)
  }

  /** Creates and returns a TensorFlow dataset, for the specified language.
    *
    * Each element of that dataset is a tuple containing:
    *   - `INT32` tensor containing the input sentence word IDs, with shape `[batchSize, maxSentenceLength]`.
    *   - `INT32` tensor containing the input sentence lengths, with shape `[batchSize]`.
    *
    * @param  language Language for which the TensorFlow dataset is constructed.
    * @return Created TensorFlow dataset.
    */
  override def toTFMonolingual(language: Language): TFMonolingualDataset = {
    checkSupportsLanguage(language)

    val batchSize = dataConfig.inferBatchSize
    val vocabTable = vocabulary(language).lookupTable()
    val bosId = vocabTable.lookup(tf.constant(dataConfig.beginOfSequenceToken)).cast(INT32)
    val eosId = vocabTable.lookup(tf.constant(dataConfig.endOfSequenceToken)).cast(INT32)

    val batchingFn = (dataset: TFMonolingualDataset) => {
      dataset.dynamicPaddedBatch(
        batchSize,
        // The first entry represents the source line rows, which are unknown-length vectors.
        // The last entry is the source row size, which is a scalar.
        (Shape(-1), Shape.scalar()),
        // We pad the source sequences with 'endSequenceToken' tokens. Though notice that we do
        // not generally need to do this since later on we will be masking out calculations past
        // the true sequence.
        (eosId, tf.zeros(INT32, Shape.scalar())))
    }

    val dataset = joinTensorDatasets(
      files(language).map(file => tf.data.TextLinesDataset(file.path.toAbsolutePath.toString())))
    val datasetBeforeBatching = dataset
        .map(o => tf.stringSplit(o.expandDims(0)).values)
        // Crop based on the maximum allowed sequence length.
        .transform(d => if (dataConfig.srcMaxLength != -1) d.map(dd => dd(0 :: dataConfig.srcMaxLength)) else d)
        // Convert the word strings to IDs. Word strings that are not in the vocabulary
        // get the lookup table's default value.
        .map(d => tf.cast(vocabTable.lookup(d), INT32))
        // Add BOS and EOS symbols.
        .map(d => tf.concatenate(Seq(bosId(NewAxis), d, eosId(NewAxis))), name = "Map/AddBosEosSymbols")
        // Add sequence lengths.
        .map(d => (d, tf.size(d, INT32)))

    batchingFn(datasetBeforeBatching)
  }

  override def toTFBilingual(
      language1: Language,
      language2: Language,
      repeat: Boolean = true,
      isEval: Boolean = false
  ): TFBilingualDataset = {
    checkSupportsLanguage(language1)
    checkSupportsLanguage(language2)

    val srcVocabTable = vocabulary(language1).lookupTable()
    val tgtVocabTable = vocabulary(language2).lookupTable()
    val batchSize = if (!isEval) dataConfig.trainBatchSize else dataConfig.evaluateBatchSize
    val actualBufferSize = if (dataConfig.bufferSize == -1L) 1000 * batchSize else dataConfig.bufferSize
    val srcBosId = srcVocabTable.lookup(tf.constant(dataConfig.beginOfSequenceToken)).cast(INT32)
    val srcEosId = srcVocabTable.lookup(tf.constant(dataConfig.endOfSequenceToken)).cast(INT32)
    val tgtBosId = tgtVocabTable.lookup(tf.constant(dataConfig.beginOfSequenceToken)).cast(INT32)
    val tgtEosId = tgtVocabTable.lookup(tf.constant(dataConfig.endOfSequenceToken)).cast(INT32)

    val srcDataset = joinTensorDatasets(
      files(language1).map(file => tf.data.TextLinesDataset(file.path.toAbsolutePath.toString())))
    val tgtDataset = joinTensorDatasets(
      files(language2).map(file => tf.data.TextLinesDataset(file.path.toAbsolutePath.toString())))

    val batchingFn = (dataset: TFBilingualDataset) => {
      dataset.dynamicPaddedBatch(
        batchSize,
        // The first three entries are the source and target line rows, which are unknown-length vectors.
        // The last two entries are the source and target row sizes, which are scalars.
        ((Shape(-1), Shape.scalar()), (Shape(-1), Shape.scalar())),
        // We pad the source and target sequences with 'endSequenceToken' tokens. Though notice that we do not
        // generally need to do this since later on we will be masking out calculations past the true sequence.
        ((srcEosId, tf.zeros(INT32, Shape.scalar().toOutput())),
            (tgtEosId, tf.zeros(INT32, Shape.scalar().toOutput()))))
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
          .shuffle(actualBufferSize)
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
          // Convert the word strings to IDs. Word strings that are not in the vocabulary
          // get the lookup table's default value.
          .map(
            d => (
                tf.cast(srcVocabTable.lookup(d._1), INT32),
                tf.cast(tgtVocabTable.lookup(d._2), INT32)),
            dataConfig.numParallelCalls, name = "Map/VocabularyLookup")
          .prefetch(actualBufferSize)
          // Add BOS and EOS symbols.
          .map(
            d => (
                tf.concatenate(Seq(srcBosId(NewAxis), d._1, srcEosId(NewAxis))),
                tf.concatenate(Seq(tgtBosId(NewAxis), d._2, tgtEosId(NewAxis)))),
            dataConfig.numParallelCalls, name = "Map/AddBosEosSymbols")
          .prefetch(actualBufferSize)
          // Add sequence lengths.
          .map(
            d => ((d._1, tf.size(d._1, INT32)), (d._2, tf.size(d._2, INT32))), dataConfig.numParallelCalls,
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

      def keyFn(element: ((Output, Output), (Output, Output))): Output = {
        // Bucket sequence  pairs based on the length of their source sequence and target sequence.
        val bucketId = tf.maximum(
          tf.truncateDivide(element._1._2, bucketWidth),
          tf.truncateDivide(element._2._2, bucketWidth))
        tf.minimum(dataConfig.numBuckets, bucketId).cast(INT64)
      }

      def reduceFn(pair: (Output, TFBilingualDataset)): TFBilingualDataset = {
        batchingFn(pair._2)
      }

      datasetBeforeBucketing.groupByWindow(keyFn, reduceFn, _ => batchSize)
    }
  }
}

object FileParallelDataset {
  def apply(
      name: String,
      vocabularies: Map[Language, Vocabulary],
      dataConfig: DataConfig,
      files: Map[Language, Seq[File]],
      fileTypes: Seq[DatasetType] = null,
      fileKeys: Seq[String] = null
  ): FileParallelDataset = {
    new FileParallelDataset(name, vocabularies, dataConfig, files, fileTypes, fileKeys)
  }
}
