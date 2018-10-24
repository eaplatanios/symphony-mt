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
      modelConfig: Model.Config,
      dataset: FileParallelDataset,
      srcLanguage: Language,
      tgtLanguage: Language,
      languages: Seq[(Language, Vocabulary)]
  ): () => InputDataset = () => {
    def createSingleInputDataset(
        srcLanguage: Output[Int],
        tgtLanguage: Output[Int],
        file: Output[String]
    ): InputDataset = {
      val endSeqToken = Tensor.fill[String](Shape())(dataConfig.endOfSequenceToken)
      tf.data.datasetFromDynamicTextFiles(file)
          .map(o => tf.stringSplit(o.expandDims(0)).values)
          // Crop based on the maximum allowed sequence length.
          .transform(d => if (dataConfig.srcMaxLength != -1) d.map(dd => dd(0 :: dataConfig.srcMaxLength)) else d)
          // Add sequence lengths.
          .map(d => (d, tf.size(d).toInt))
          .paddedBatch(
            batchSize = dataConfig.inferBatchSize,
            // The first entry represents the source line rows, which are unknown-length vectors.
            // The last entry is the source row size, which is a scalar.
            paddedShapes = (Shape(-1), Shape()),
            // We pad the source sequences with 'endSequenceToken' tokens. Though notice that we do
            // not generally need to do this since later on we will be masking out calculations past
            // the true sequence.
            paddingValues = Some((endSeqToken, Tensor.zeros[Int](Shape()))))
          .map(d => (srcLanguage, tgtLanguage, (d._1, d._2)))
    }

    val languageIds = languages.map(_._1).zipWithIndex.toMap
    val files = dataset.files(srcLanguage).map(_.path.toAbsolutePath.toString()): Tensor[String]
    val numFiles = files.size

    tf.data.datasetFromTensorSlices(files)
        .map(
          function = d => (tf.constant(languageIds(srcLanguage)), tf.constant(languageIds(tgtLanguage)), d),
          name = "AddLanguageIDs")
        .shuffle(numFiles)
        .interleave(
          function = d => createSingleInputDataset(d._1, d._2, d._3),
          cycleLength = numFiles,
          name = "Interleave")
  }

  def createTrainDataset(
      dataConfig: DataConfig,
      modelConfig: Model.Config,
      datasets: Seq[FileParallelDataset],
      languages: Seq[(Language, Vocabulary)],
      includeIdentityTranslations: Boolean = false,
      repeat: Boolean = true,
      isEval: Boolean = false,
      languagePairs: Option[Set[(Language, Language)]] = None
  ): () => TrainDataset = () => {
    val languageIds = languages.map(_._1).zipWithIndex.toMap
    val bufferSize = if (dataConfig.bufferSize == -1L) 1024L else dataConfig.bufferSize

    val filteredDatasets = datasets
        .map(_.filterLanguages(languageIds.keys.toSeq: _*))
        .filter(_.nonEmpty)
        .flatMap(d => {
          var currentLanguagePairs = d.languagePairs(includeIdentityTranslations)
          if (dataConfig.parallelPortion == 0.0f)
            currentLanguagePairs = currentLanguagePairs.filter(p => p._1 == p._2)
          val providedLanguagePairs = languagePairs match {
            case Some(pairs) if includeIdentityTranslations => pairs.flatMap(p => Seq(p, (p._1, p._1), (p._2, p._2)))
            case Some(pairs) => pairs
            case None => currentLanguagePairs
          }
          providedLanguagePairs.intersect(currentLanguagePairs).map(_ -> d)
        })
        .groupBy(_._1)
        .mapValues(_.map(_._2))
    val maxNumFiles = filteredDatasets.map(d => d._2.map(_.files(d._1._1).size).sum).max
    val numParallelFiles = filteredDatasets.map(d => d._2.map(_.files(d._1._1).size).sum).sum

    // Each element in `filesDataset` is a tuple: (srcLanguage, tgtLanguage, srcFile, tgtFile).
    val filesDataset = filteredDatasets
        .map {
          case ((srcLanguage, tgtLanguage), parallelDatasets) =>
            val srcFiles = parallelDatasets.flatMap(_.files(srcLanguage))
            val tgtFiles = parallelDatasets.flatMap(_.files(tgtLanguage))
            val srcLengths = srcFiles.map(_.lineIterator(StandardCharsets.UTF_8).size)
            val tgtLengths = tgtFiles.map(_.lineIterator(StandardCharsets.UTF_8).size)
            val srcLanguageDataset = tf.data.datasetFromTensors(languageIds(srcLanguage): Tensor[Int])
            val tgtLanguageDataset = tf.data.datasetFromTensors(languageIds(tgtLanguage): Tensor[Int])
            val srcFilesDataset = tf.data.datasetFromTensors(srcFiles.map(_.path.toAbsolutePath.toString()): Tensor[String])
            val tgtFilesDataset = tf.data.datasetFromTensors(tgtFiles.map(_.path.toAbsolutePath.toString()): Tensor[String])
            val srcLengthsDataset = tf.data.datasetFromTensors(srcLengths: Tensor[Int])
            val tgtLengthsDataset = tf.data.datasetFromTensors(tgtLengths: Tensor[Int])
            srcLanguageDataset.zip(tgtLanguageDataset)
                .zip(srcFilesDataset.zip(tgtFilesDataset))
                .zip(srcLengthsDataset.zip(tgtLengthsDataset))
                .map(d => (d._1._1._1, d._1._1._2, d._1._2._1, d._1._2._2, d._2._1, d._2._2), name = "AddLanguageIDs")
        }.reduce((d1, d2) => d1.concatenateWith(d2))

    val parallelDatasetCreator: (Output[Int], Output[Int], Output[String], Output[String], Output[Int], Output[Int]) => TrainDataset =
      createSingleParallelDataset(dataConfig, modelConfig, repeat, isEval)

    filesDataset
        .shuffle(filteredDatasets.size)
        .interleave(
          function = d => {
            val (srcLanguage, tgtLanguage, srcFiles, tgtFiles, srcLengths, tgtLengths) = d
            val srcLanguageDataset = tf.data.datasetFromOutputs(srcLanguage).repeat()
            val tgtLanguageDataset = tf.data.datasetFromOutputs(tgtLanguage).repeat()
            val srcFilesDataset = tf.data.datasetFromOutputSlices(srcFiles)
            val tgtFilesDataset = tf.data.datasetFromOutputSlices(tgtFiles)
            val srcLengthsDataset = tf.data.datasetFromOutputSlices(srcLengths)
            val tgtLengthsDataset = tf.data.datasetFromOutputSlices(tgtLengths)
            srcLanguageDataset.zip(tgtLanguageDataset)
                .zip(srcFilesDataset.zip(tgtFilesDataset))
                .zip(srcLengthsDataset.zip(tgtLengthsDataset))
                .map(d => (d._1._1._1, d._1._1._2, d._1._2._1, d._1._2._2, d._2._1, d._2._2), name = "AddLanguageIDs")
                .shuffle(maxNumFiles)
                .interleave(
                  function = d => parallelDatasetCreator(d._1, d._2, d._3, d._4, d._5, d._6),
                  cycleLength = maxNumFiles,
                  name = "FilesInterleave")
          },
          cycleLength = numParallelFiles,
          numParallelCalls = numParallelFiles,
          name = "LanguagePairsInterleave")
  }

  def createEvalDatasets(
      dataConfig: DataConfig,
      modelConfig: Model.Config,
      datasets: Seq[(String, FileParallelDataset, Float)],
      languages: Seq[(Language, Vocabulary)],
      languagePairs: Option[Set[(Language, Language)]] = None
  ): Seq[(String, () => TrainDataset)] = {
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
        .sortBy(d => (d._1._1, (d._1._2._1.abbreviation, d._1._2._2.abbreviation)))
        .map {
          case ((name, (srcLanguage, tgtLanguage), parallelPortion), dataset) =>
            val datasetName = s"$name/${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"
            (datasetName, () => tf.nameScope(datasetName) {
              createTrainDataset(
                dataConfig.copy(parallelPortion = parallelPortion), modelConfig, Seq(dataset), languages,
                includeIdentityTranslations = false, repeat = false, isEval = true,
                languagePairs = Some(Set((srcLanguage, tgtLanguage))))()
            })
        }
  }

  private def createSingleParallelDataset(
      dataConfig: DataConfig,
      modelConfig: Model.Config,
      repeat: Boolean,
      isEval: Boolean
  )(
      srcLanguage: Output[Int],
      tgtLanguage: Output[Int],
      srcFile: Output[String],
      tgtFile: Output[String],
      srcLength: Output[Int],
      tgtLength: Output[Int]
  ): TrainDataset = {
    val batchSize = if (!isEval) dataConfig.trainBatchSize else dataConfig.evaluateBatchSize
    val bufferSize = if (dataConfig.bufferSize == -1L) 64L * batchSize else dataConfig.bufferSize

    val srcLanguageDataset = tf.data.datasetFromOutputs(srcLanguage).repeat()
    val tgtLanguageDataset = tf.data.datasetFromOutputs(tgtLanguage).repeat()
    val srcDataset = tf.data.datasetFromDynamicTextFiles(srcFile)
    val tgtDataset = tf.data.datasetFromDynamicTextFiles(tgtFile)

    // TODO: We currently do not use `tgtLength`, but it may be useful for invalid dataset checks.

    val numParallel = tf.cond(
      predicate = srcLanguage.equal(tgtLanguage),
      trueFn = () => srcLength.toLong,
      falseFn = () => (srcLength.toFloat * dataConfig.parallelPortion).floor.toLong)

    val datasetBeforeBucketing =
      srcLanguageDataset.zip(tgtLanguageDataset).zip(srcDataset.zip(tgtDataset)
          .take(numParallel))
          .shard(dataConfig.numShards, dataConfig.shardIndex)
          .transform(d => if (repeat) d.repeat() else d)
          .transform(d => if (!isEval) d.shuffle(bufferSize) else d)
          // Tokenize by splitting on white spaces.
          .map(
            d => (d._1, (tf.stringSplit(d._2._1(NewAxis)).values, tf.stringSplit(d._2._2(NewAxis)).values)),
            name = "Map/StringSplit")
          // Filter zero length input sequences and sequences exceeding the maximum length.
          .filter(d => tf.logicalAnd(tf.size(d._2._1) > 0, tf.size(d._2._2) > 0), "Filter/NonZeroLength")
          // Crop based on the maximum allowed sequence lengths.
          .transform(d => {
            if (!isEval && dataConfig.srcMaxLength != -1 && dataConfig.tgtMaxLength != -1) {
              d.map(
                dd => (dd._1, (dd._2._1(0 :: dataConfig.srcMaxLength), dd._2._2(0 :: dataConfig.tgtMaxLength))),
                name = "Map/MaxLength")
            } else if (!isEval && dataConfig.srcMaxLength != -1) {
              d.map(
                dd => (dd._1, (dd._2._1(0 :: dataConfig.srcMaxLength), dd._2._2)),
                name = "Map/MaxLength")
            } else if (!isEval && dataConfig.tgtMaxLength != -1) {
              d.map(
                dd => (dd._1, (dd._2._1, dd._2._2(0 :: dataConfig.tgtMaxLength))),
                name = "Map/MaxLength")
            } else {
              d
            }
          })
          .prefetch(bufferSize)
          // Add sequence lengths.
          .map(d => (
              /* Language pair */ (d._1._1, d._1._2),
              /* Source sentences */ (d._2._1, tf.size(d._2._1).toInt),
              /* Target sentences */ (d._2._2, tf.size(d._2._2).toInt)),
            name = "Map/AddLengths")
          .prefetch(bufferSize)

    val batchingFn = (dataset: SentencePairsDataset) => {
      val zero = Tensor.zeros[Int](Shape())
      val endSeqToken = Tensor.fill[String](Shape())(dataConfig.endOfSequenceToken)
      dataset.paddedBatch(
        batchSize = batchSize,
        // The first three entries are the source and target line rows, which are unknown-length vectors.
        // The last two entries are the source and target row sizes, which are scalars.
        paddedShapes = ((Shape(), Shape()), (Shape(-1), Shape()), (Shape(-1), Shape())),
        // We pad the source and target sequences with 'endSequenceToken' tokens. Though notice that we do not
        // generally need to do this since later on we will be masking out calculations past the true sequence.
        paddingValues = Some(((zero, zero), (endSeqToken, zero), (endSeqToken, zero))))
    }

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

        def keyFn(element: SentencePairs[String]): Output[Long] = {
          // Bucket sequence  pairs based on the length of their source sequence and target sequence.
          val bucketId = tf.maximum(
            tf.truncateDivide(element._2._2, bucketWidth), // Source sentence lengths
            tf.truncateDivide(element._3._2, bucketWidth)) // Target sentence lengths
          tf.minimum(dataConfig.numBuckets, bucketId).toLong
        }

        def reduceFn(pair: (Output[Long], SentencePairsDataset)): SentencePairsDataset = {
          batchingFn(pair._2)
        }

        datasetBeforeBucketing.groupByWindow(keyFn, reduceFn, _ => batchSize)
      }
    }

    parallelDataset
        .map(d => (
            (/* Source language */ d._1._1(0), /* Target language */ d._1._2(0), /* Source sentences */ d._2),
            /* Target sentences */ d._3),
          name = "AddLanguageIDs")
        .prefetch(bufferSize)
  }
}
