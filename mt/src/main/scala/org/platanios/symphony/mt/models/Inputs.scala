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
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._

import java.nio.charset.StandardCharsets

// TODO: Sample files with probability proportional to their size.

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
      languages: Seq[(Language, Vocabulary)]
  ): () => TFInputDataset = () => {
    val languageIds = languages.map(_._1).zipWithIndex.toMap

    val files: Tensor = dataset.files(srcLanguage).map(_.path.toAbsolutePath.toString())
    val numFiles = files.size

    val inputDatasetCreator: (Output, Output, Output) => TFInputDataset = createSingleInputDataset(dataConfig, config)

    tf.data.TensorSlicesDataset(files)
        .map(
          d => (tf.constant(languageIds(srcLanguage)), tf.constant(languageIds(tgtLanguage)), d),
          name = "AddLanguageIDs")
        .shuffle(numFiles)
        .parallelInterleave(d => inputDatasetCreator(d._1, d._2, d._3), cycleLength = numFiles, name = "Interleave")
        .asInstanceOf[TFInputDataset]
  }

  def createTrainDataset(
      dataConfig: DataConfig,
      config: Model.Config,
      datasets: Seq[FileParallelDataset],
      languages: Seq[(Language, Vocabulary)],
      includeBackTranslations: Boolean = false,
      repeat: Boolean = true,
      isEval: Boolean = false,
      languagePairs: Option[Set[(Language, Language)]] = None
  ): () => TFTrainDataset = () => {
    val languageIds = languages.map(_._1).zipWithIndex.toMap
    val bufferSize = if (dataConfig.bufferSize == -1L) 1024L else dataConfig.bufferSize

    val filteredDatasets = datasets
        .map(_.filterLanguages(languageIds.keys.toSeq: _*))
        .filter(_.nonEmpty)
        .flatMap(d => {
          var currentLanguagePairs = d.languagePairs(includeBackTranslations)
          if (dataConfig.parallelPortion == 0.0f)
            currentLanguagePairs = currentLanguagePairs.filter(p => p._1 == p._2)
          languagePairs.getOrElse(currentLanguagePairs).intersect(currentLanguagePairs).map(_ -> d)
        })
    val numParallelFiles = filteredDatasets.map(d => d._2.files(d._1._1).size).sum

    // Each element in `filesDataset` is a tuple: (srcLanguage, tgtLanguage, srcFile, tgtFile).
    val filesDataset = filteredDatasets
        .map {
          case ((srcLanguage, tgtLanguage), dataset) =>
            val srcFiles = dataset.files(srcLanguage)
            val tgtFiles = dataset.files(tgtLanguage)
            val srcLengths = srcFiles.map(_.lineIterator(StandardCharsets.UTF_8).size)
            val tgtLengths = tgtFiles.map(_.lineIterator(StandardCharsets.UTF_8).size)
            val srcLanguageDataset = tf.data.TensorDataset(languageIds(srcLanguage): Tensor).repeat()
            val tgtLanguageDataset = tf.data.TensorDataset(languageIds(tgtLanguage): Tensor).repeat()
            val srcFilesDataset = tf.data.TensorSlicesDataset(srcFiles.map(_.path.toAbsolutePath.toString()): Tensor)
            val tgtFilesDataset = tf.data.TensorSlicesDataset(tgtFiles.map(_.path.toAbsolutePath.toString()): Tensor)
            val srcLengthsDataset = tf.data.TensorSlicesDataset(srcLengths: Tensor)
            val tgtLengthsDataset = tf.data.TensorSlicesDataset(tgtLengths: Tensor)
            srcLanguageDataset.zip(tgtLanguageDataset)
                .zip(srcFilesDataset.zip(tgtFilesDataset))
                .zip(srcLengthsDataset.zip(tgtLengthsDataset)).map(
              d => (d._1._1._1, d._1._1._2, d._1._2._1, d._1._2._2, d._2._1, d._2._2),
              name = "AddLanguageIDs")
        }.reduce((d1, d2) => d1.concatenate(d2))

    val parallelDatasetCreator: (Output, Output, Output, Output, Output, Output) => TFTrainDataset =
      createSingleParallelDataset(dataConfig, config, repeat, isEval)

    filesDataset
        .shuffle(numParallelFiles)
        .parallelInterleave(
          d => parallelDatasetCreator(d._1, d._2, d._3, d._4, d._5, d._6), cycleLength = numParallelFiles,
          sloppy = true, bufferOutputElements = bufferSize, prefetchInputElements = numParallelFiles,
          name = "Interleave")
  }

  def createEvalDatasets(
      dataConfig: DataConfig,
      config: Model.Config,
      datasets: Seq[(String, FileParallelDataset, Float)],
      languages: Seq[(Language, Vocabulary)],
      languagePairs: Option[Set[(Language, Language)]] = None
  ): Seq[(String, () => TFTrainDataset)] = {
    datasets
        .map(d => (d._1, d._2.filterLanguages(languages.map(_._1): _*), d._3))
        .flatMap(d => {
          val currentLanguagePairs = d._2.languagePairs()
          languagePairs.getOrElse(currentLanguagePairs)
              .intersect(currentLanguagePairs)
              .map(l => (d._1, l, d._3) -> d._2)
        })
        .toMap
        .toSeq
        .map {
          case ((name, (srcLanguage, tgtLanguage), parallelPortion), dataset) =>
            val datasetName = s"$name/${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"
            (datasetName, () => tf.createWithNameScope(datasetName) {
              createTrainDataset(
                dataConfig.copy(parallelPortion = parallelPortion), config, Seq(dataset), languages,
                includeBackTranslations = false, repeat = false, isEval = true,
                languagePairs = Some(Set((srcLanguage, tgtLanguage))))()
            })
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
      config: Model.Config
  )(srcLanguage: Output, tgtLanguage: Output, file: Output): TFInputDataset = {
    val batchSize = dataConfig.inferBatchSize

    val batchingFn = (dataset: TFSentencesDataset) => {
      dataset.dynamicPaddedBatch(
        batchSize,
        // The first entry represents the source line rows, which are unknown-length vectors.
        // The last entry is the source row size, which is a scalar.
        (Shape(-1), Shape.scalar()),
        // We pad the source sequences with 'endSequenceToken' tokens. Though notice that we do
        // not generally need to do this since later on we will be masking out calculations past
        // the true sequence.
        (tf.constant(dataConfig.endOfSequenceToken), tf.zeros(INT32, Shape.scalar())))
    }

    val dataset = tf.data.DynamicTextLinesDataset(file)

    val datasetBeforeBatching = dataset
        .map(o => tf.stringSplit(o.expandDims(0)).values)
        // Crop based on the maximum allowed sequence length.
        .transform(d => if (dataConfig.srcMaxLength != -1) d.map(dd => dd(0 :: dataConfig.srcMaxLength)) else d)
        // Add sequence lengths.
        .map(d => (d, tf.size(d, INT32)))

    batchingFn(datasetBeforeBatching)
        .map(d => (srcLanguage, tgtLanguage, d._1, d._2))
  }

  private[this] def createSingleParallelDataset(
      dataConfig: DataConfig,
      config: Model.Config,
      repeat: Boolean,
      isEval: Boolean
  )(
      srcLanguage: Output,
      tgtLanguage: Output,
      srcFile: Output,
      tgtFile: Output,
      srcLength: Output,
      tgtLength: Output
  ): TFTrainDataset = {
    val batchSize = if (!isEval) dataConfig.trainBatchSize else dataConfig.evaluateBatchSize
    val bufferSize = if (dataConfig.bufferSize == -1L) 64L * batchSize else dataConfig.bufferSize

    val srcLanguageDataset = tf.data.OutputDataset(srcLanguage).repeat()
    val tgtLanguageDataset = tf.data.OutputDataset(tgtLanguage).repeat()
    val srcDataset = tf.data.DynamicTextLinesDataset(srcFile)
    val tgtDataset = tf.data.DynamicTextLinesDataset(tgtFile)

    val batchingFn = (dataset: TFSentencePairsDataset) => {
      dataset.dynamicPaddedBatch(
        batchSize,
        // The first three entries are the source and target line rows, which are unknown-length vectors.
        // The last two entries are the source and target row sizes, which are scalars.
        ((Shape(), Shape()), ((Shape(-1), Shape()), (Shape(-1), Shape()))),
        // We pad the source and target sequences with 'endSequenceToken' tokens. Though notice that we do not
        // generally need to do this since later on we will be masking out calculations past the true sequence.
        ((tf.zeros(INT32, Shape()), tf.zeros(INT32, Shape())),
            ((tf.constant(dataConfig.endOfSequenceToken), tf.zeros(INT32, Shape.scalar().toOutput())),
                (tf.constant(dataConfig.endOfSequenceToken), tf.zeros(INT32, Shape.scalar().toOutput())))))
    }

    // TODO: We currently do not use `tgtLength`, but it may be useful for invalid dataset checks.

    val datasetBeforeBucketing =
      srcLanguageDataset.zip(tgtLanguageDataset)
          .zip(srcDataset.zip(tgtDataset)
              .take(tf.cond(srcLanguage.equal(tgtLanguage),
                () => srcLength.cast(INT64),
                () => (srcLength * dataConfig.parallelPortion).floor.cast(INT64))))
          .shard(dataConfig.numShards, dataConfig.shardIndex)
          .transform(d => {
            if (repeat)
              d.repeat()
            else
              d
          })
          .transform(d => {
            if (!isEval)
              d.shuffle(bufferSize)
            else
              d
          })
          // Tokenize by splitting on white spaces.
          .map(
            d => (d._1, (tf.stringSplit(d._2._1(NewAxis)).values, tf.stringSplit(d._2._2(NewAxis)).values)),
            name = "Map/StringSplit")
          // Filter zero length input sequences and sequences exceeding the maximum length.
          .filter(d => tf.logicalAnd(tf.size(d._2._1) > 0, tf.size(d._2._2) > 0), "Filter/NonZeroLength")
          // Crop based on the maximum allowed sequence lengths.
          .transform(d => {
            if (!isEval && dataConfig.srcMaxLength != -1 && dataConfig.tgtMaxLength != -1)
              d.map(
                dd => (dd._1, (dd._2._1(0 :: dataConfig.srcMaxLength), dd._2._2(0 :: dataConfig.tgtMaxLength))),
                dataConfig.numParallelCalls, name = "Map/MaxLength")
            else if (!isEval && dataConfig.srcMaxLength != -1)
              d.map(
                dd => (dd._1, (dd._2._1(0 :: dataConfig.srcMaxLength), dd._2._2)),
                dataConfig.numParallelCalls, name = "Map/MaxLength")
            else if (!isEval && dataConfig.tgtMaxLength != -1)
              d.map(
                dd => (dd._1, (dd._2._1, dd._2._2(0 :: dataConfig.tgtMaxLength))),
                dataConfig.numParallelCalls, name = "Map/MaxLength")
            else
              d
          })
          .prefetch(bufferSize)
          // Add sequence lengths.
          .map(
            d => (d._1, ((d._2._1, tf.size(d._2._1, INT32)), (d._2._2, tf.size(d._2._2, INT32)))),
            dataConfig.numParallelCalls, name = "Map/AddLengths")
          .prefetch(bufferSize)

    val parallelDataset = {
      if (dataConfig.numBuckets == 1 || isEval) {
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

        def keyFn(element: ((Output, Output), ((Output, Output), (Output, Output)))): Output = {
          // Bucket sequence  pairs based on the length of their source sequence and target sequence.
          val bucketId = tf.maximum(
            tf.truncateDivide(element._2._1._2, bucketWidth),
            tf.truncateDivide(element._2._2._2, bucketWidth))
          tf.minimum(dataConfig.numBuckets, bucketId).cast(INT64)
        }

        def reduceFn(pair: (Output, TFSentencePairsDataset)): TFSentencePairsDataset = {
          batchingFn(pair._2)
        }

        datasetBeforeBucketing.groupByWindow(keyFn, reduceFn, _ => batchSize)
      }
    }

    parallelDataset
        .map(
          d => ((d._1._1(0), d._1._2(0), d._2._1._1, d._2._1._2), (d._2._2._1, d._2._2._2)),
          dataConfig.numParallelCalls, name = "AddLanguageIDs")
        .prefetch(bufferSize)
        .asInstanceOf[TFTrainDataset]
  }
}
