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
import org.platanios.symphony.mt.utilities.{CompressedFiles, MutableFile}

import better.files._
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.io.IOException
import java.net.URL
import java.nio.file.Path

import scala.collection.mutable

// TODO: [DATA] Placed processed files in a different directory.

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
        // TODO: The following file moves are hacky and non-generic (they only apply to the WMT-16 dataset).
        if (newFile.name.endsWith(s"-src.$src") || newFile.name.endsWith(s"-ref.$src")) {
          val renamedFile = newFile.sibling(newFile.name.dropRight(5 + src.length) + s".$src")
          if (renamedFile.notExists)
            newFile.copyTo(renamedFile)
          newFile = renamedFile
        } else if (newFile.name.endsWith(s"-src.$tgt") || newFile.name.endsWith(s"-ref.$tgt")) {
          val renamedFile = newFile.sibling(newFile.name.dropRight(5 + tgt.length) + s".$tgt")
          if (renamedFile.notExists)
            newFile.copyTo(renamedFile)
          newFile = renamedFile
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
  def load(): FileParallelDataset = ParallelDatasetLoader.load(Seq(this))._1.head
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
            logger.info(
              s"│${"═" * numBars}${" " * (10 - numBars)}│ " +
                  s"%${contentLength.toString.length}s / $contentLength bytes downloaded.".format(progress))
            progressLogTime = time
          }
        })
        outputStream.close()
        logger.info(
          s"│${"═" * 10}│ %${contentLength.toString.length}s / $contentLength bytes downloaded.".format(progress))
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

  def load(
      loaders: Seq[ParallelDatasetLoader],
      workingDir: Option[File] = None
  ): (Seq[FileParallelDataset], Seq[(Language, Vocabulary)]) = {
    // Collect all files.
    val files = loaders.map(loader => {
      var srcFiles = Seq.empty[MutableFile]
      var tgtFiles = Seq.empty[MutableFile]
      var fileTypes = Seq.empty[DatasetType]
      var fileKeys = Seq.empty[String]
      DatasetType.types.foreach(datasetType => {
        val corpora = loader.corpora(datasetType)

        // Tokenize source and target files.
        srcFiles ++= corpora.map(_._2).map(f => {
          loader.dataConfig.loaderTokenizer.tokenizeCorpus(
            f, loader.srcLanguage, loader.dataConfig.loaderBufferSize)
        }).map(MutableFile(_))
        tgtFiles ++= corpora.map(_._3).map(f => {
          loader.dataConfig.loaderTokenizer.tokenizeCorpus(
            f, loader.tgtLanguage, loader.dataConfig.loaderBufferSize)
        }).map(MutableFile(_))

        fileTypes ++= Seq.fill(corpora.length)(datasetType)
        fileKeys ++= corpora.map(_._1)
      })

      // TODO: [DATA] Only clean the training data.

      // Clean the corpora.
      val dataCleaning = loader.dataConfig.loaderCleaner
      srcFiles.zip(tgtFiles).foreach {
        case (srcFile, tgtFile) =>
          val cleaned = dataCleaning.processCorporaPair(srcFile.get, tgtFile.get, loader.dataConfig.loaderBufferSize)
          srcFile.set(cleaned._1)
          tgtFile.set(cleaned._2)
      }

      (srcFiles, tgtFiles, fileTypes, fileKeys)
    })

    // Generate vocabularies, if necessary.
    val vocabularies = mutable.Map.empty[Language, mutable.ListBuffer[File]]
    loaders.foreach(loader => {
      vocabularies.getOrElseUpdate(loader.srcLanguage, mutable.ListBuffer.empty).append(loader.vocabularies._1: _*)
      vocabularies.getOrElseUpdate(loader.tgtLanguage, mutable.ListBuffer.empty).append(loader.vocabularies._2: _*)
    })
    val vocabDir = workingDir.getOrElse(File(loaders.head.dataConfig.workingDir)) / "vocabularies"
    val vocabulary = vocabularies.toMap.map {
      case (l, v) =>
        val vocabFilename = loaders.head.dataConfig.loaderVocab.filename(l)
        loaders.head.dataConfig.loaderVocab match {
          case NoVocabulary => l -> null // TODO: Avoid using nulls.
          case GeneratedVocabulary(generator) =>
            l -> {
              val tokenizedFiles = loaders.zip(files).flatMap {
                case (loader, f) if loader.srcLanguage == l => f._1
                case (loader, f) if loader.tgtLanguage == l => f._2
                case _ => Seq.empty
              }
              generator.generate(l, tokenizedFiles, vocabDir)
              generator.getVocabulary(l, vocabDir)
            }
          case MergedVocabularies if v.lengthCompare(1) == 0 => l -> Vocabulary(v.head)
          case MergedVocabularies if v.nonEmpty =>
            val vocabFile = vocabDir.createChild(vocabFilename, createParents = true)
            val writer = newWriter(vocabFile)
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

    val datasets = loaders.zip(files).map {
      case (loader, (srcFiles, tgtFiles, fileTypes, fileKeys)) =>
        val groupedFiles = Map(loader.srcLanguage -> srcFiles.map(_.get), loader.tgtLanguage -> tgtFiles.map(_.get))
        val filteredVocabulary = vocabulary.filterKeys(l => l == loader.srcLanguage || l == loader.tgtLanguage)
        FileParallelDataset(loader.name, filteredVocabulary, loader.dataConfig, groupedFiles, fileTypes, fileKeys)
    }

    (datasets, vocabulary.toSeq)
  }
}
