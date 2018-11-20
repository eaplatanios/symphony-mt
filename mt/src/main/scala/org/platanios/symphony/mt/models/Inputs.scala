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
import org.platanios.symphony.mt.config.TrainingConfig
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.data.processors.TFRecordsConverter
import org.platanios.symphony.mt.models.curriculum.Curriculum
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.Parsing.FixedLengthFeature

// TODO: Sample files with probability proportional to their size.

/**
  * @author Emmanouil Antonios Platanios
  */
object Inputs {
  def createInputDataset(
      dataConfig: DataConfig,
      dataset: FileParallelDataset,
      srcLanguage: Language,
      tgtLanguage: Language,
      languages: Seq[(Language, Vocabulary)],
      useTFRecords: Boolean = true
  ): () => tf.data.Dataset[SentencesWithLanguagePair[String]] = () => {
    def createSingleInputDataset(
        srcLanguage: Output[Int],
        tgtLanguage: Output[Int],
        file: Output[String]
    ): tf.data.Dataset[SentencesWithLanguagePair[String]] = {
      val endSeqToken = Tensor.fill[String](Shape())(Vocabulary.END_OF_SEQUENCE_TOKEN)

      var dataset = {
        if (useTFRecords) {
          tf.data.datasetFromDynamicTFRecordFiles(file, bufferSize = dataConfig.loaderBufferSize)
              .map(parseTFRecord, name = "Map/ParseExample")
        } else {
          tf.data.datasetFromDynamicTextFiles(file)
              .map(o => tf.stringSplit(o.expandDims(0)).values)
              // Add sequence lengths.
              .map(d => (d, tf.size(d).toInt))
        }
      }

      // Crop based on the maximum allowed sequence length.
      if (dataConfig.srcMaxLength != -1)
        dataset = dataset.map(d => (d._1(0 :: dataConfig.srcMaxLength), tf.minimum(d._2, dataConfig.srcMaxLength)))

      dataset = dataset.paddedBatch(
        batchSize = dataConfig.inferBatchSize,
        // The first entry represents the source line rows, which are unknown-length vectors.
        // The last entry is the source row size, which is a scalar.
        paddedShapes = (Shape(-1), Shape()),
        // We pad the source sequences with 'endSequenceToken' tokens. Though notice that we do
        // not generally need to do this since later on we will be masking out calculations past
        // the true sequence.
        paddingValues = Some((endSeqToken, Tensor.zeros[Int](Shape()))))

      dataset.map(d => (srcLanguage, tgtLanguage, (d._1, d._2)))
    }

    val languageIds = languages.map(_._1).zipWithIndex.toMap
    val files = dataset.files(srcLanguage).map(file => {
      if (useTFRecords)
        TFRecordsConverter.process(file, srcLanguage)
      else
        file
    }).map(_.path.toAbsolutePath.toString()): Tensor[String]

    tf.data.datasetFromTensorSlices(files)
        .map(
          function = d => (tf.constant(languageIds(srcLanguage)), tf.constant(languageIds(tgtLanguage)), d),
          name = "AddLanguageIDs")
        .shuffle(files.size)
        .interleave(
          function = d => createSingleInputDataset(d._1, d._2, d._3),
          cycleLength = files.size,
          name = "Interleave")
  }

  def createTrainDataset(
      dataConfig: DataConfig,
      trainingConfig: TrainingConfig,
      datasets: Seq[FileParallelDataset],
      languages: Seq[(Language, Vocabulary)],
      includeIdentityTranslations: Boolean = false,
      cache: Boolean = false,
      repeat: Boolean = true,
      isEval: Boolean = false,
      languagePairs: Option[Set[(Language, Language)]] = None
  ): () => tf.data.Dataset[(SentencesWithLanguagePair[String], Sentences[String])] = () => {
    tf.device("/CPU:0") {
      val languageIds = languages.map(_._1).zipWithIndex.toMap
      val filteredDatasets = datasets
          .map(_.filterLanguages(languageIds.keys.toSeq: _*))
          .filter(_.nonEmpty)
          .flatMap(d => {
            val currentLanguagePairs = d.languagePairs(includeIdentityTranslations)
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
              val srcFiles = parallelDatasets.flatMap(_.files(srcLanguage)).map(f => TFRecordsConverter.process(f, srcLanguage))
              val tgtFiles = parallelDatasets.flatMap(_.files(tgtLanguage)).map(f => TFRecordsConverter.process(f, tgtLanguage))
              val srcLanguageDataset = tf.data.datasetFromTensors(languageIds(srcLanguage): Tensor[Int])
              val tgtLanguageDataset = tf.data.datasetFromTensors(languageIds(tgtLanguage): Tensor[Int])
              val srcFilesDataset = tf.data.datasetFromTensors(srcFiles.map(_.path.toAbsolutePath.toString()): Tensor[String])
              val tgtFilesDataset = tf.data.datasetFromTensors(tgtFiles.map(_.path.toAbsolutePath.toString()): Tensor[String])
              srcLanguageDataset.zip(tgtLanguageDataset)
                  .zip(srcFilesDataset.zip(tgtFilesDataset))
                  .map(d => (d._1._1, d._1._2, d._2._1, d._2._2), name = "AddLanguageIDs")
          }.reduce((d1, d2) => d1.concatenateWith(d2))

      val datasetCreator = createSingleParallelDataset(dataConfig, trainingConfig, cache, repeat, isEval)(_, _, _, _)

      filesDataset
          .shuffle(filteredDatasets.size)
          .interleave(
            function = d => {
              val (srcLanguage, tgtLanguage, srcFiles, tgtFiles) = d
              val srcLanguageDataset = tf.data.datasetFromOutputs(srcLanguage).repeat()
              val tgtLanguageDataset = tf.data.datasetFromOutputs(tgtLanguage).repeat()
              val srcFilesDataset = tf.data.datasetFromOutputSlices(srcFiles)
              val tgtFilesDataset = tf.data.datasetFromOutputSlices(tgtFiles)
              srcLanguageDataset.zip(tgtLanguageDataset)
                  .zip(srcFilesDataset.zip(tgtFilesDataset))
                  .map(d => (d._1._1, d._1._2, d._2._1, d._2._2), name = "AddLanguageIDs")
                  .shuffle(maxNumFiles)
                  .interleave(
                    function = d => datasetCreator(d._1, d._2, d._3, d._4),
                    cycleLength = maxNumFiles,
                    name = "FilesInterleave")
            },
            cycleLength = numParallelFiles,
            numParallelCalls = maxNumFiles,
            name = "LanguagePairsInterleave")
    }
  }

  def createEvalDatasets(
      dataConfig: DataConfig,
      trainingConfig: TrainingConfig,
      datasets: Seq[(String, FileParallelDataset)],
      languages: Seq[(Language, Vocabulary)],
      languagePairs: Option[Set[(Language, Language)]] = None
  ): Seq[(String, () => tf.data.Dataset[(SentencesWithLanguagePair[String], Sentences[String])])] = {
    datasets
        .map(d => (d._1, d._2.filterLanguages(languages.map(_._1): _*)))
        .flatMap(d => {
          val currentLanguagePairs = d._2.languagePairs()
          languagePairs.getOrElse(currentLanguagePairs)
              .intersect(currentLanguagePairs)
              .map(l => (d._1, l) -> d._2)
        })
        .toMap
        .toSeq
        .sortBy(d => (d._1._1, (d._1._2._1.abbreviation, d._1._2._2.abbreviation)))
        .map {
          case ((name, (srcLanguage, tgtLanguage)), dataset) =>
            val datasetName = s"$name/${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"
            (datasetName, () => tf.nameScope(datasetName) {
              createTrainDataset(
                dataConfig,
                trainingConfig = trainingConfig.copy(curriculum = Curriculum.none),
                Seq(dataset), languages,
                includeIdentityTranslations = false, cache = false, repeat = false, isEval = true,
                languagePairs = Some(Set((srcLanguage, tgtLanguage))))()
            })
        }
  }

  private def createSingleParallelDataset(
      dataConfig: DataConfig,
      trainingConfig: TrainingConfig,
      cache: Boolean,
      repeat: Boolean,
      isEval: Boolean
  )(
      srcLanguage: Output[Int],
      tgtLanguage: Output[Int],
      srcFile: Output[String],
      tgtFile: Output[String]
  ): tf.data.Dataset[(SentencesWithLanguagePair[String], Sentences[String])] = {
    val batchSize = if (!isEval) dataConfig.trainBatchSize else dataConfig.evalBatchSize
    val shuffleBufferSize = if (dataConfig.shuffleBufferSize == -1L) 10L * batchSize else dataConfig.shuffleBufferSize

    val srcLanguageDataset = tf.data.datasetFromOutputs(srcLanguage).repeat()
    val tgtLanguageDataset = tf.data.datasetFromOutputs(tgtLanguage).repeat()

    val srcDataset = tf.data.datasetFromDynamicTFRecordFiles(srcFile, bufferSize = dataConfig.loaderBufferSize)
    val tgtDataset = tf.data.datasetFromDynamicTFRecordFiles(tgtFile, bufferSize = dataConfig.loaderBufferSize)

    var datasetBeforeBucketing = srcLanguageDataset.zip(tgtLanguageDataset).zip(srcDataset.zip(tgtDataset))
        .map(d => (d._1, parseTFRecord(d._2._1), parseTFRecord(d._2._2)), name = "Map/ParseExample")

    // Crop based on the maximum allowed sequence lengths.
    if (!isEval && dataConfig.srcMaxLength != -1 && dataConfig.tgtMaxLength != -1) {
      datasetBeforeBucketing = datasetBeforeBucketing.map(d => (
          d._1,
          (d._2._1(0 :: dataConfig.srcMaxLength), tf.minimum(d._2._2, dataConfig.srcMaxLength)),
          (d._3._1(0 :: dataConfig.tgtMaxLength), tf.minimum(d._3._2, dataConfig.tgtMaxLength))),
        name = "Map/MaxLength")
    } else if (!isEval && dataConfig.srcMaxLength != -1) {
      datasetBeforeBucketing = datasetBeforeBucketing.map(d =>
        (d._1, (d._2._1(0 :: dataConfig.srcMaxLength), tf.minimum(d._2._2, dataConfig.srcMaxLength)), d._3),
        name = "Map/MaxLength")
    } else if (!isEval && dataConfig.tgtMaxLength != -1) {
      datasetBeforeBucketing = datasetBeforeBucketing.map(d =>
        (d._1, d._2, (d._3._1(0 :: dataConfig.tgtMaxLength), tf.minimum(d._3._2, dataConfig.tgtMaxLength))),
        name = "Map/MaxLength")
    }

    if (cache)
      datasetBeforeBucketing = datasetBeforeBucketing.cache("")

    trainingConfig.curriculum.samplesFilter match {
      case Some(samplesFilter) => datasetBeforeBucketing = datasetBeforeBucketing.filter(samplesFilter)
      case None => ()
    }

    if (!isEval && repeat)
      datasetBeforeBucketing = datasetBeforeBucketing.shuffleAndRepeat(shuffleBufferSize)
    else if (!isEval)
      datasetBeforeBucketing = datasetBeforeBucketing.shuffle(shuffleBufferSize)

    val batchingFn = (dataset: tf.data.Dataset[SentencePairs[String]]) => {
      val zero = Tensor.zeros[Int](Shape())
      val endSeqToken = Tensor.fill[String](Shape())(Vocabulary.END_OF_SEQUENCE_TOKEN)
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

        def reduceFn(
            pair: (Output[Long], tf.data.Dataset[SentencePairs[String]])
        ): tf.data.Dataset[SentencePairs[String]] = {
          batchingFn(pair._2)
        }

        def windowSizeFn(key: Output[Long]): Output[Long] = {
          val providedBatchSize = tf.constant[Long](batchSize)
          if (dataConfig.bucketAdaptedBatchSize) {
            val one = tf.constant[Long](1L)
            tf.maximum(tf.truncateDivide(providedBatchSize, key + one), one)
          } else {
            providedBatchSize
          }
        }

        datasetBeforeBucketing.groupByWindow(keyFn, reduceFn, windowSizeFn)
      }
    }

    parallelDataset
        .map(d => (
            ( /* Source language */ d._1._1(0), /* Target language */ d._1._2(0), /* Source sentences */ d._2),
            /* Target sentences */ d._3),
          name = "AddLanguageIDs")
        .prefetch(dataConfig.numPrefetchedBatches)
  }

  private def parseTFRecord(serialized: Output[String]): (Output[String], Output[Int]) = {
    val parsedExample = tf.parseSingleExample(
      serialized = serialized,
      features = (
          FixedLengthFeature[String](key = "sentence", shape = Shape(-1)),
          FixedLengthFeature[Long](key = "length", shape = Shape())),
      name = "ParseExample")
    (parsedExample._1, parsedExample._2.toInt)
  }
}
