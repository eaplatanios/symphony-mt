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
import org.platanios.symphony.mt.data.utilities.CompressedFiles

import better.files._
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.io.IOException
import java.net.URL
import java.nio.file.Path

/** Parallel dataset used for machine translation experiments.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class Dataset(
    val srcLanguage: Language,
    val tgtLanguage: Language,
    val dataConfig: DataConfig
)(
    val downloadsDir: Path = dataConfig.loaderWorkingDir
) {
  def name: String

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
      Dataset.maybeDownload(path, url, dataConfig.loaderBufferSize)
      path
    })
  }

  /** Extracted files (if needed) from `downloadedFiles`. */
  protected val extractedFiles: Seq[File] = {
    if (!dataConfig.loaderExtractTGZ) {
      downloadedFiles.flatMap(_.listRecursively.filter(_.isRegularFile))
    } else {
      Dataset.logger.info(s"$name - Extracting any downloaded archives.")
      val files = downloadedFiles.flatMap(
        Dataset.maybeExtractTGZ(_, dataConfig.loaderBufferSize)
            .listRecursively
            .filter(_.isRegularFile))
      Dataset.logger.info(s"$name - Extracted any downloaded archives.")
      files
    }
  }

  /** Preprocessed files after converting extracted SGM files to normal text files, and (optionally) tokenizing. */
  protected val preprocessedFiles: Seq[File] = {
    Dataset.logger.info(s"$name - Preprocessing any downloaded files.")
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
    Dataset.logger.info(s"$name - Preprocessed any downloaded files.")
    files
  }

  /** Returns all the train corpora (tuples containing name, source file, and target file) of this dataset. */
  def trainCorpora: Seq[(String, File, File)] = Seq.empty

  /** Returns all the dev corpora (tuples containing name, source file, and target file) of this dataset. */
  def devCorpora: Seq[(String, File, File)] = Seq.empty

  /** Returns all the test corpora (tuples containing name, source file, and target file) of this dataset. */
  def testCorpora: Seq[(String, File, File)] = Seq.empty

  /** Returns the source and the target vocabulary of this dataset. */
  def vocabularies: Option[(File, File)] = None

  /** Returns the files included in this dataset, grouped based on their role. */
  def load(): LoadedDataset = {
    var files = LoadedDataset.GroupedFiles(
      srcLanguage, tgtLanguage, dataConfig, trainCorpora, devCorpora, testCorpora, vocabularies)
    dataConfig.loaderSentenceLengthBounds.foreach {
      case (minLength, maxLength) =>
        files = files.copy(trainCorpora = files.trainCorpora.map(files => {
          // TODO: [DATA] This is a hacky way of checking for the clean corpus files.
          val corpusFile = files._2.sibling(files._2.nameWithoutExtension(includeAll = false))
          val cleanCorpusFile = files._2.sibling(corpusFile.name + ".clean")
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
          (files._1, srcCleanCorpusFile, tgtCleanCorpusFile)
        }))
    }
    LoadedDataset(dataConfig, Set(files))
  }
}

object Dataset {
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
