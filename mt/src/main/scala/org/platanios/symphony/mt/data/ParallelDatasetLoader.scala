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
import org.platanios.symphony.mt.utilities.CompressedFiles
import better.files._
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.io.{BufferedWriter, IOException}
import java.net.URL
import java.nio.file.Path

import org.platanios.symphony.mt.vocabulary.Vocabulary

/** Parallel dataset used for machine translation experiments.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class ParallelDatasetLoader(val srcLanguage: Language, val tgtLanguage: Language) {
  def name: String

  def dataConfig: DataConfig

  def downloadsDir: Path = dataConfig.workingDir

  protected def src: String = srcLanguage.abbreviation
  protected def tgt: String = tgtLanguage.abbreviation

  // Clone the Moses repository, if necessary.
  val mosesDecoder: Utilities.MosesDecoder = Utilities.MosesDecoder(File(downloadsDir) / "moses")
  if (!mosesDecoder.exists)
    mosesDecoder.cloneRepository()

  /** Sequence of files to download as part of this dataset. */
  def filesToDownload: Seq[String]

  /** Downloaded files listed in `filesToDownload`. */
  protected val downloadedFiles: Seq[File] = {
    filesToDownload.map(url => {
      val (_, filename) = url.splitAt(url.lastIndexOf('/') + 1)
      val path = File(downloadsDir) / filename
      ParallelDatasetLoader.maybeDownload(path, url, dataConfig.loaderBufferSize)
      path
    })
  }

  /** Extracted files (if needed) from `downloadedFiles`. */
  protected val extractedFiles: Seq[File] = {
    if (!dataConfig.loaderExtractTGZ) {
      downloadedFiles.flatMap(_.listRecursively.filter(_.isRegularFile))
    } else {
      ParallelDatasetLoader.logger.info(s"$name - Extracting any downloaded archives.")
      val files = downloadedFiles.flatMap(
        ParallelDatasetLoader.maybeExtractTGZ(_, dataConfig.loaderBufferSize)
            .listRecursively
            .filter(_.isRegularFile))
      ParallelDatasetLoader.logger.info(s"$name - Extracted any downloaded archives.")
      files
    }
  }

  /** Preprocessed files after converting extracted SGM files to normal text files, and (optionally) tokenizing. */
  protected val preprocessedFiles: Seq[File] = {
    ParallelDatasetLoader.logger.info(s"$name - Preprocessing any downloaded files.")
    val files = extractedFiles.map(file => {
      var newFile = file
      if (!newFile.name.startsWith(".")) {
        if (dataConfig.loaderConvertSGMToText && file.extension().contains(".sgm")) {
          newFile = file.sibling(file.nameWithoutExtension(includeAll = false))
          if (newFile.notExists)
            mosesDecoder.sgmToText(file, newFile)
        }
        if (dataConfig.loaderTokenize && !newFile.name.contains(".tok")) {
          // TODO: The language passed to the tokenizer is "computed" in a non-standardized way.
          val tokenizedFile = newFile.sibling(
            newFile.nameWithoutExtension(includeAll = false) + ".tok" + newFile.extension.getOrElse(""))
          if (tokenizedFile.notExists) {
            val exitCode = mosesDecoder.tokenize(
              newFile, tokenizedFile, tokenizedFile.extension(includeDot = false).getOrElse(""))
            if (exitCode != 0)
              newFile.copyTo(tokenizedFile)
          }
          newFile = tokenizedFile
        }
      }
      newFile
    })
    ParallelDatasetLoader.logger.info(s"$name - Preprocessed any downloaded files.")
    files
  }

  /** Returns all the corpora (tuples containing name, source file, and target file) of this dataset type. */
  def corpora(datasetType: DatasetType): Seq[(String, File, File)] = Seq.empty

  /** Returns the source and the target language vocabularies of this dataset. */
  def vocabularies: (Seq[File], Seq[File]) = (Seq.empty, Seq.empty)

  /** Returns the files included in this dataset, grouped based on their role. */
  def load(): FileParallelDataset = {
    // Collect all files.
    var srcFiles = Seq.empty[File]
    var tgtFiles = Seq.empty[File]
    var fileTypes = Seq.empty[DatasetType]
    var fileKeys = Seq.empty[String]
    DatasetType.types.foreach(datasetType => {
      val typeCorpora = corpora(datasetType)
      srcFiles ++= typeCorpora.map(_._2)
      tgtFiles ++= typeCorpora.map(_._3)
      fileTypes ++= Seq.fill(typeCorpora.length)(datasetType)
      fileKeys ++= typeCorpora.map(_._1)
    })

    // Clean the corpora, if necessary.
    dataConfig.loaderSentenceLengthBounds.foreach {
      case (minLength, maxLength) =>
        val cleanedFiles = srcFiles.map(files => {
          // TODO: [DATA] This is a hacky way of checking for the clean corpus files.
          val corpusFile = files.sibling(files.nameWithoutExtension(includeAll = false))
          val cleanCorpusFile = files.sibling(corpusFile.name + ".clean")
          val srcCleanCorpusFile = corpusFile.sibling(cleanCorpusFile.name + s".$src")
          val tgtCleanCorpusFile = corpusFile.sibling(cleanCorpusFile.name + s".$tgt")
          if (srcCleanCorpusFile.notExists || tgtCleanCorpusFile.notExists) {
            val exitCode = mosesDecoder.cleanCorpus(
              corpusFile, cleanCorpusFile, src, tgt, minLength, maxLength)
            if (exitCode != 0) {
              corpusFile.sibling(corpusFile.name + s".$src").copyTo(srcCleanCorpusFile)
              corpusFile.sibling(corpusFile.name + s".$tgt").copyTo(tgtCleanCorpusFile)
            }
          }
          (srcCleanCorpusFile, tgtCleanCorpusFile)
        }).unzip
        srcFiles = cleanedFiles._1
        tgtFiles = cleanedFiles._2
    }

    // Generate vocabularies, if necessary.
    val workingDir = File(dataConfig.workingDir)
    val vocabulary = Map(srcLanguage -> vocabularies._1, tgtLanguage -> vocabularies._2).map {
      case (l, v) =>
        dataConfig.loaderVocab match {
          case GeneratedVocabulary(generator) =>
            l -> {
              val tokenizedFiles = if (l == srcLanguage) srcFiles else tgtFiles
              val vocabFile = workingDir / s"vocab.${l.abbreviation}"
              if (vocabFile.notExists) {
                ParallelDatasetLoader.logger.info(s"Generating vocabulary file for $l.")
                generator.generate(tokenizedFiles, vocabFile)
                ParallelDatasetLoader.logger.info(s"Generated vocabulary file for $l.")
              }
              Vocabulary(vocabFile)
            }
          case MergedVocabularies if v.lengthCompare(1) == 0 => l -> Vocabulary(v.head)
          case MergedVocabularies if v.nonEmpty =>
            val vocabFile = workingDir.createChild(s"vocab.${l.abbreviation}", createParents = true)
            val writer = new BufferedWriter(vocabFile.newPrintWriter(), dataConfig.loaderBufferSize)
            v.toStream
                .flatMap(_.lineIterator).toSet
                .filter(_ != "")
                .foreach(word => writer.write(word + "\n"))
            writer.flush()
            writer.close()
            l -> Vocabulary(vocabFile)
          case MergedVocabularies if v.isEmpty =>
            throw new IllegalArgumentException("No existing vocabularies found to merge.")
        }
    }

    val files = Map(srcLanguage -> srcFiles, tgtLanguage -> tgtFiles)
    FileParallelDataset(name, vocabulary, dataConfig, files, fileTypes, fileKeys)
  }
}

object ParallelDatasetLoader {
  private[data] val logger = Logger(LoggerFactory.getLogger("Dataset"))

  def maybeDownload(file: File, url: String, bufferSize: Int = 8192): Boolean = {
    if (file.exists) {
      false
    } else {
      try {
        logger.info(s"Downloading file '$url'.")
        file.parent.createDirectories()
        val connection = new URL(url).openConnection()
        val contentLength = connection.getContentLengthLong
        val inputStream = connection.getInputStream
        val outputStream = file.newOutputStream
        val buffer = new Array[Byte](bufferSize)
        var progress = 0L
        var progressLogTime = System.currentTimeMillis
        Stream.continually(inputStream.read(buffer)).takeWhile(_ != -1).foreach(numBytes => {
          outputStream.write(buffer, 0, numBytes)
          progress += numBytes
          val time = System.currentTimeMillis
          if (time - progressLogTime >= 1e4) {
            val numBars = Math.floorDiv(10 * progress, contentLength).toInt
            logger.info(s"[${"=" * numBars}${" " * (10 - numBars)}] $progress / $contentLength bytes downloaded.")
            progressLogTime = time
          }
        })
        outputStream.close()
        logger.info(s"Downloaded file '$url'.")
        true
      } catch {
        case e: IOException =>
          logger.error(s"Could not download file '$url'", e)
          throw e
      }
    }
  }

  def maybeExtractTGZ(file: File, bufferSize: Int = 8192): File = {
    if (file.extension(includeAll = true).exists(ext => ext == ".tgz" || ext == ".tar.gz")) {
      val processedFile = file.sibling(file.nameWithoutExtension(includeAll = true))
      if (processedFile.notExists)
        CompressedFiles.decompressTGZ(file, processedFile, bufferSize)
      processedFile
    } else {
      file
    }
  }
}
