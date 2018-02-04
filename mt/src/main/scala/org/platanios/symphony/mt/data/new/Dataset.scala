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

package org.platanios.symphony.mt.data.`new`

import org.platanios.symphony.mt.data.Utilities
import org.platanios.symphony.mt.data.utilities.CompressedFiles

import better.files._
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.io.IOException
import java.net.URL
import java.nio.file.Path

/** Parallel dataset used for machine translation experiments.
  *
  * @param  workingDir  Working directory for this dataset.
  * @param  bufferSize  Buffer size to use when downloading and processing files.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class Dataset(
    val workingDir: Path,
    val bufferSize: Int = 8192,
    val tokenize: Boolean = false
)(
    val downloadsDir: Path = workingDir.resolve("downloads")
) {
  // Clone the Moses repository, if necessary.
  val mosesDecoder: Utilities.MosesDecoder = Utilities.MosesDecoder(File(downloadsDir) / "moses")
  if (!mosesDecoder.exists)
    mosesDecoder.cloneRepository()

  /** Sequence of files to download as part of this dataset. */
  val filesToDownload: Seq[String]

  /** Downloaded files listed in `filesToDownload`. */
  protected val downloadedFiles: Seq[File] = {
    filesToDownload.map(url => {
      val (_, filename) = url.splitAt(url.lastIndexOf('/') + 1)
      val path = File(downloadsDir) / filename
      Dataset.maybeDownload(path, url, bufferSize)
      path
    })
  }

  /** Extracted files (if needed) from `downloadedFiles`. */
  protected val extractedFiles: Seq[File] = {
    downloadedFiles.flatMap(Dataset.maybeExtractTGZ(_, bufferSize).listRecursively.filter(_.isRegularFile))
  }

  /** Preprocessed files after converting extracted SGM files to normal text files, and (optionally) tokenizing. */
  protected val preprocessedFiles: Seq[File] = {
    extractedFiles.map(file => {
      var newFile = file
      if (file.extension().contains(".sgm")) {
        newFile = file.changeExtensionTo("")
        mosesDecoder.sgmToText(file, newFile)
      }
      if (tokenize) {
        // TODO: The language passed to the tokenizer is "computed" in a non-standardized way.
        val tokenizedFile = newFile.renameTo(newFile.nameWithoutExtension + ".tok" + file.extension)
        mosesDecoder.tokenize(newFile, tokenizedFile, newFile.extension(includeDot = false).get)
        newFile = tokenizedFile
      }
      newFile
    })
  }

  /** Grouped files included in this dataset. */
  val groupedFiles: Dataset.GroupedFiles
}

object Dataset {
  private[data] val logger = Logger(LoggerFactory.getLogger("Dataset"))

  case class GroupedFiles(
      trainCorpora: Seq[(String, File, File)] = Seq.empty,
      devCorpora: Seq[(String, File, File)] = Seq.empty,
      testCorpora: Seq[(String, File, File)] = Seq.empty,
      vocabularies: Option[(File, File)] = None)

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
      CompressedFiles.decompressTGZ(file, processedFile, bufferSize)
      processedFile
    } else {
      file
    }
  }
}
