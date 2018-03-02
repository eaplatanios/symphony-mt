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

package org.platanios.symphony.mt.models

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.vocabulary.{Vocabularies, Vocabulary}
import org.platanios.tensorflow.api._

/**
  * @author Emmanouil Antonios Platanios
  */
object Inputs {
  def createInputDataset(
      dataConfig: DataConfig,
      config: Model.Config,
      dataset: FileParallelDataset,
      srcLanguage: Language,
      tgtLanguage: Language,
      languages: Map[Language, Vocabulary]
  ): () => TFInputDataset = () => {
    val languageIds = languages.keys.zipWithIndex.toMap
    // TODO: Recreating the vocabularies here may be inefficient.
    val vocabularies = Vocabularies(languages, config.embeddingsSize)

    val files: Tensor = dataset.files(srcLanguage).map(_.path.toAbsolutePath.toString())
    val numFiles = files.size

    val inputDatasetCreator: (Output, Output, Output) => TFInputDataset =
      createSingleInputDataset(dataConfig, config, vocabularies)

    tf.data.TensorSlicesDataset(files)
        .map(
          d => (tf.constant(languageIds(srcLanguage)), tf.constant(languageIds(tgtLanguage)), d),
          name = "AddLanguageIDs")
        .shuffle(numFiles)
        .parallelInterleave(d => inputDatasetCreator(d._1, d._2, d._3), cycleLength = numFiles)
        .asInstanceOf[TFInputDataset]
  }

  def createTrainDataset(
      dataConfig: DataConfig,
      config: Model.Config,
      datasets: Seq[FileParallelDataset],
      languages: Map[Language, Vocabulary],
      repeat: Boolean = true,
      isEval: Boolean = false
  ): () => TFTrainDataset = () => {
    val languageIds = languages.keys.zipWithIndex.toMap
    // TODO: Recreating the vocabularies here may be inefficient.
    val vocabularies = Vocabularies(languages, config.embeddingsSize)
    val bufferSize = if (dataConfig.bufferSize == -1L) 1024L else dataConfig.bufferSize

    val filteredDatasets = datasets.map(_.filterLanguages(languageIds.keys.toSeq: _*))
    val numParallelFiles = filteredDatasets.map(_.languagePairs().size).sum // TODO: Not correct.

    // Each element in `filesDataset` is a tuple: (srcLanguage, tgtLanguage, srcFile, tgtFile).
    val filesDataset = filteredDatasets
        .flatMap(d => d.languagePairs().map(_ -> d))
        .map {
          case ((srcLanguage, tgtLanguage), dataset) =>
            val srcFiles: Tensor = dataset.files(srcLanguage).map(_.path.toAbsolutePath.toString())
            val tgtFiles: Tensor = dataset.files(tgtLanguage).map(_.path.toAbsolutePath.toString())
            val srcFilesDataset = tf.data.TensorSlicesDataset(srcFiles)
            val tgtFilesDataset = tf.data.TensorSlicesDataset(tgtFiles)
            srcFilesDataset.zip(tgtFilesDataset).map(
              d => (tf.constant(languageIds(srcLanguage)), tf.constant(languageIds(tgtLanguage)), d._1, d._2),
              name = "AddLanguageIDs")
        }.reduce((d1, d2) => d1.concatenate(d2))

    val parallelDatasetCreator: (Output, Output, Output, Output) => TFTrainDataset =
      createSingleParallelDataset(dataConfig, config, vocabularies, repeat, isEval)

    filesDataset
        .shuffle(numParallelFiles)
        .parallelInterleave(d => parallelDatasetCreator(d._1, d._2, d._3, d._4), cycleLength = numParallelFiles)
        .prefetch(bufferSize)
  }

  def createEvalDatasets(
      dataConfig: DataConfig,
      config: Model.Config,
      datasets: Seq[(String, FileParallelDataset)],
      languages: Map[Language, Vocabulary]
  ): Seq[(String, () => TFTrainDataset)] = {
    datasets
        .map(d => (d._1, d._2.filterLanguages(languages.keys.toSeq: _*)))
        .flatMap(d => d._2.languagePairs().map(l => (d._1, l) -> d._2))
        .map {
          case ((name, (srcLanguage, tgtLanguage)), dataset) =>
            (s"$name/${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}",
                createTrainDataset(dataConfig, config, Seq(dataset), languages, repeat = false, isEval = true))
        }
  }

  /** Creates and returns a TensorFlow dataset, for the specified language.
    *
    * Each element of that dataset is a tuple containing:
    *   - `INT32` tensor containing the input sentence word IDs, with shape `[batchSize, maxSentenceLength]`.
    *   - `INT32` tensor containing the input sentence lengths, with shape `[batchSize]`.
    *
    *
    *
    *
    * @return Created TensorFlow dataset.
    */
  private[this] def createSingleInputDataset(
      dataConfig: DataConfig,
      config: Model.Config,
      vocabularies: Vocabularies
  )(srcLanguage: Output, tgtLanguage: Output, file: Output): TFInputDataset = {
    val batchSize = dataConfig.inferBatchSize
    val srcVocabLookup = vocabularies.lookupTable(srcLanguage)
    val eosId = srcVocabLookup(tf.constant(dataConfig.endOfSequenceToken)).cast(INT32)

    val batchingFn = (dataset: TFSentencesDataset) => {
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

    val dataset = tf.data.DynamicTextLinesDataset(file)

    val datasetBeforeBatching = dataset
        .map(o => tf.stringSplit(o.expandDims(0)).values)
        // Crop based on the maximum allowed sequence length.
        .transform(d => if (dataConfig.srcMaxLength != -1) d.map(dd => dd(0 :: dataConfig.srcMaxLength)) else d)
        // Convert the word strings to IDs. Word strings that are not in the vocabulary
        // get the lookup table's default value.
        .map(d => tf.cast(srcVocabLookup(d), INT32))
        // Add sequence lengths.
        .map(d => (d, tf.size(d, INT32)))

    batchingFn(datasetBeforeBatching)
        .map(d => (srcLanguage, tgtLanguage, d._1, d._2))
  }

  private[this] def createSingleParallelDataset(
      dataConfig: DataConfig,
      config: Model.Config,
      vocabularies: Vocabularies,
      repeat: Boolean,
      isEval: Boolean
  )(
      srcLanguage: Output,
      tgtLanguage: Output,
      srcFile: Output,
      tgtFile: Output
  ): TFTrainDataset = {
    val batchSize = if (!isEval) dataConfig.trainBatchSize else dataConfig.evaluateBatchSize
    val bufferSize = if (dataConfig.bufferSize == -1L) 1024L * batchSize else dataConfig.bufferSize

    val srcVocabLookup = vocabularies.lookupTable(srcLanguage)
    val tgtVocabLookup = vocabularies.lookupTable(tgtLanguage)
    val srcEosId = srcVocabLookup(tf.constant(dataConfig.endOfSequenceToken)).cast(INT32)
    val tgtEosId = tgtVocabLookup(tf.constant(dataConfig.endOfSequenceToken)).cast(INT32)

    val srcDataset = tf.data.DynamicTextLinesDataset(srcFile)
    val tgtDataset = tf.data.DynamicTextLinesDataset(tgtFile)

    val batchingFn = (dataset: TFSentencePairsDataset) => {
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
              d.repeat()
            else
              d
          })
          .shuffle(bufferSize)
          // Tokenize by splitting on white spaces.
          .map(
            d => (tf.stringSplit(d._1(NewAxis)).values, tf.stringSplit(d._2(NewAxis)).values),
            name = "Map/StringSplit")
              // Filter zero length input sequences and sequences exceeding the maximum length.
              .filter(d => tf.logicalAnd(tf.size(d._1) > 0, tf.size(d._2) > 0), "Filter/NonZeroLength")
              // Crop based on the maximum allowed sequence lengths.
              .transform(d => {
            if (dataConfig.srcMaxLength != -1 && dataConfig.tgtMaxLength != -1)
              d.map(
                dd => (dd._1(0 :: dataConfig.srcMaxLength), dd._2(0 :: dataConfig.tgtMaxLength)),
                dataConfig.numParallelCalls, name = "Map/MaxLength")
            else if (dataConfig.srcMaxLength != -1)
              d.map(
                dd => (dd._1(0 :: dataConfig.srcMaxLength), dd._2),
                dataConfig.numParallelCalls, name = "Map/MaxLength")
            else if (dataConfig.tgtMaxLength != -1)
              d.map(
                dd => (dd._1, dd._2(0 :: dataConfig.tgtMaxLength)),
                dataConfig.numParallelCalls, name = "Map/MaxLength")
            else
              d
          })
          // Convert the word strings to IDs. Word strings that are not in the vocabulary
          // get the lookup table's default value.
          .map(
            d => (
                tf.cast(srcVocabLookup(d._1), INT32),
                tf.cast(tgtVocabLookup(d._2), INT32)),
            dataConfig.numParallelCalls, name = "Map/VocabularyLookup")
              // Add sequence lengths.
              .map(
            d => ((d._1, tf.size(d._1, INT32)), (d._2, tf.size(d._2, INT32))), dataConfig.numParallelCalls,
            name = "Map/AddLengths")

    val parallelDataset = {
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

        def reduceFn(pair: (Output, TFSentencePairsDataset)): TFSentencePairsDataset = {
          batchingFn(pair._2)
        }

        datasetBeforeBucketing.groupByWindow(keyFn, reduceFn, _ => batchSize)
      }
    }

    parallelDataset
        .map(d => ((srcLanguage, tgtLanguage, d._1._1, d._1._2), dataConfig.numParallelCalls, d._2))
        .prefetch(bufferSize)
        .asInstanceOf[TFTrainDataset]
  }
}
