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
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.io.data.TextLinesDataset

import better.files._

import java.io.BufferedWriter

import scala.collection.immutable.Traversable

/**
  * @author Emmanouil Antonios Platanios
  */
class LoadedDataset private[LoadedDataset] (
    protected val dataConfig: DataConfig,
    protected val datasets: Map[(Language, Language), LoadedDataset.GroupedFiles]
) {
  val workingDir: File = File(dataConfig.loaderWorkingDir)

  def languagePairs: Iterable[(Language, Language)] = datasets.keys

  def files(srcLanguage: Language, tgtLanguage: Language): LoadedDataset.GroupedFiles = {
    datasets.getOrElse((srcLanguage, tgtLanguage), datasets((tgtLanguage, srcLanguage)).reversed)
  }
}

object LoadedDataset {
  def apply(dataConfig: DataConfig, files: Traversable[GroupedFiles]): LoadedDataset = {
    val workingDir = File(dataConfig.loaderWorkingDir)
    val vocabularies = {
      (files.groupBy(_.srcLanguage).mapValues(_.filter(_.vocabularies.isDefined).map(_.vocabularies.get._1)) ++
          files.groupBy(_.tgtLanguage).mapValues(_.filter(_.vocabularies.isDefined).map(_.vocabularies.get._2)))
          .map {
            case (language, vocabFiles) =>
              if (vocabFiles.size == 1) {
                language -> vocabFiles.head
              } else if (vocabFiles.isEmpty || !dataConfig.loaderMergeVocabs) {
                language -> {
                  val tokenizedFiles =
                    files.filter(_.srcLanguage == language).flatMap(_.trainCorpora.map(_._2)) ++
                        files.filter(_.tgtLanguage == language).flatMap(_.trainCorpora.map(_._3))
                  val vocabFile = workingDir / s"vocab.${language.abbreviation}"
                  if (vocabFile.notExists) {
                    Dataset.logger.info(s"Creating vocabulary file for $language.")
                    Vocabulary.createVocabFile(
                      tokenizedFiles.toSeq, vocabFile,
                      dataConfig.vocabSizeThreshold, dataConfig.vocabCountThreshold,
                      dataConfig.loaderBufferSize)
                    Dataset.logger.info(s"Created vocabulary file for $language.")
                  }
                  vocabFile
                }
              } else {
                val vocabFile = workingDir.createChild(s"vocab.${language.abbreviation}", createParents = true)
                val writer = new BufferedWriter(vocabFile.newPrintWriter(), dataConfig.loaderBufferSize)
                vocabFiles.toStream
                    .flatMap(_.lineIterator).toSet
                    .filter(_ != "")
                    .foreach(word => writer.write(word + "\n"))
                writer.flush()
                writer.close()
                language -> vocabFile
              }
          }
    }
    new LoadedDataset(
      dataConfig,
      files
          .groupBy(dataset => (dataset.srcLanguage, dataset.tgtLanguage))
          .mapValues(_.reduce((dataset1, dataset2) => {
            GroupedFiles(
              srcLanguage = dataset1.srcLanguage,
              tgtLanguage = dataset1.tgtLanguage,
              dataConfig = dataConfig,
              trainCorpora = dataset1.trainCorpora ++ dataset2.trainCorpora,
              devCorpora = dataset1.devCorpora ++ dataset2.devCorpora,
              testCorpora = dataset1.testCorpora ++ dataset2.testCorpora,
              vocabularies = Some((vocabularies(dataset1.srcLanguage), vocabularies(dataset1.tgtLanguage))))
          })))
  }

  def merge(dataConfig: DataConfig, datasets: Traversable[LoadedDataset]): LoadedDataset = {
    LoadedDataset(dataConfig, datasets.flatMap(_.datasets.values))
  }

  private[data] def joinDatasets(datasets: Seq[MTTextLinesDataset]): MTTextLinesDataset = {
    datasets.reduce((d1, d2) => d1.concatenate(d2))
  }

  case class GroupedFiles(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataConfig: DataConfig,
      trainCorpora: Seq[(String, File, File)] = Seq.empty,
      devCorpora: Seq[(String, File, File)] = Seq.empty,
      testCorpora: Seq[(String, File, File)] = Seq.empty,
      vocabularies: Option[(File, File)] = None
  ) {
    private[GroupedFiles] val _vocabularies: Option[(Vocabulary, Vocabulary)] = {
      vocabularies.map(files => (Vocabulary(files._1), Vocabulary(files._2)))
    }

    lazy val srcVocab: Vocabulary = {
      val files = if (vocabularies.isDefined) this else withNewVocab()
      files._vocabularies.get._1
    }

    lazy val tgtVocab: Vocabulary = {
      val files = if (vocabularies.isDefined) this else withNewVocab()
      files._vocabularies.get._2
    }

    def reversed: GroupedFiles = {
      GroupedFiles(
        tgtLanguage, srcLanguage, dataConfig,
        trainCorpora.map(f => (f._1, f._3, f._2)),
        devCorpora.map(f => (f._1, f._3, f._2)),
        testCorpora.map(f => (f._1, f._3, f._2)),
        vocabularies.map(f => (f._2, f._1)))
    }

    def withNewVocab(): GroupedFiles = {
      val workingDir = File(dataConfig.loaderWorkingDir)
      val srcFiles = trainCorpora.map(_._2) ++ devCorpora.map(_._2) ++ testCorpora.map(_._2)
      val tgtFiles = trainCorpora.map(_._3) ++ devCorpora.map(_._3) ++ testCorpora.map(_._3)
      val srcVocab = workingDir / s"vocab.${srcLanguage.abbreviation}"
      val tgtVocab = workingDir / s"vocab.${tgtLanguage.abbreviation}"
      if (srcVocab.notExists) {
        Dataset.logger.info(s"Creating vocabulary file for ${srcLanguage.abbreviation}.")
        Vocabulary.createVocabFile(
          srcFiles, srcVocab,
          dataConfig.vocabSizeThreshold, dataConfig.vocabCountThreshold,
          dataConfig.loaderBufferSize)
        Dataset.logger.info(s"Created vocabulary file for ${srcLanguage.abbreviation}.")
      }
      if (tgtVocab.notExists) {
        Dataset.logger.info(s"Creating vocabulary file for ${tgtLanguage.abbreviation}.")
        Vocabulary.createVocabFile(
          tgtFiles, tgtVocab,
          dataConfig.vocabSizeThreshold, dataConfig.vocabCountThreshold,
          dataConfig.loaderBufferSize)
        Dataset.logger.info(s"Created vocabulary file for ${tgtLanguage.abbreviation}.")
      }
      Dataset.logger.info("Created vocabulary files.")
      copy(vocabularies = Some((srcVocab, tgtVocab)))
    }

    def createInferDataset(
        datasetType: DatasetType,
        batchSize: Int,
        dataConfig: DataConfig = dataConfig
    ): MTInferDataset = {
      val files = if (vocabularies.isDefined) this else withNewVocab()
      val corpora = datasetType match {
        case TRAIN_DATASET => files.trainCorpora
        case DEV_DATASET => files.devCorpora
        case TEST_DATASET => files.testCorpora
      }
      val srcDatasets = corpora.map(_._2).map(file => TextLinesDataset(file.path.toAbsolutePath.toString()))
      val srcDataset = joinDatasets(srcDatasets)
      val srcVocabularyTable = files._vocabularies.get._1.lookupTable()
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
        datasetType: DatasetType,
        batchSize: Int,
        repeat: Boolean = true,
        dataConfig: DataConfig = dataConfig
    ): MTTrainDataset = {
      val files = if (vocabularies.isDefined) this else withNewVocab()
      val corpora = datasetType match {
        case TRAIN_DATASET => files.trainCorpora
        case DEV_DATASET => files.devCorpora
        case TEST_DATASET => files.testCorpora
      }
      val srcTrainDatasets = corpora.map(_._2).map(file => TextLinesDataset(file.path.toAbsolutePath.toString()))
      val tgtTrainDatasets = corpora.map(_._3).map(file => TextLinesDataset(file.path.toAbsolutePath.toString()))
      val srcDataset = joinDatasets(srcTrainDatasets)
      val tgtDataset = joinDatasets(tgtTrainDatasets)
      val srcVocabularyTable = files._vocabularies.get._1.lookupTable()
      val tgtVocabularyTable = files._vocabularies.get._2.lookupTable()
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
}
