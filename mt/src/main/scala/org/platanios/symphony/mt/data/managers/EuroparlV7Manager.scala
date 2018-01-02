/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.symphony.mt.data.managers

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.Language._
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.data.utilities.CompressedFiles

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.{Files, Path}

/**
  * @author Emmanouil Antonios Platanios
  */
case class EuroparlV7Manager(
    workingDir: Path,
    srcLanguage: Language,
    tgtLanguage: Language
) extends Manager(workingDir.resolve("europarl-v7")) {
  require(
    EuroparlV7Manager.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the Europarl v7 data manager.")

  val src: String = srcLanguage.abbreviation
  val tgt: String = tgtLanguage.abbreviation

  val name: String = s"$src-$tgt"

  private[this] val reversed: Boolean = {
    EuroparlV7Manager.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  private[this] val corpusArchiveFile: String = if (reversed) s"$tgt-$src" else s"$src-$tgt"

  private[this] val corpusFilenamePrefix: String = {
    s"europarl-v7.${if (reversed) s"$tgt-$src" else s"$src-$tgt"}"
  }

  override def download(bufferSize: Int = 8192): ParallelDataset = {
    // Download and decompress the data, if necessary.
    val archivePathPrefix = path.resolve(corpusArchiveFile)
    val archivePath = path.resolve(s"$corpusArchiveFile.tgz")
    val srcTrainCorpus = archivePathPrefix.resolve(s"$corpusFilenamePrefix.$src")
    val tgtTrainCorpus = archivePathPrefix.resolve(s"$corpusFilenamePrefix.$tgt")

    if (!Files.exists(archivePathPrefix)) {
      Manager.maybeDownload(archivePath, s"${EuroparlV7Manager.url}/$corpusArchiveFile.tgz", bufferSize)
      CompressedFiles.decompressTGZ(archivePath, archivePathPrefix, bufferSize)
    }

    downloadUpdatedArchives(corpusArchiveFile, path, Seq(corpusArchiveFile), bufferSize)
    ParallelDataset(
      workingDir = path,
      trainCorpora = Map(srcLanguage -> Seq(srcTrainCorpus), tgtLanguage -> Seq(tgtTrainCorpus)))
  }

  protected def downloadUpdatedArchives(
      archiveName: String, path: Path, archives: Seq[String], bufferSize: Int = 8192
  ): Unit = {
    val archivePath = path.resolve(archiveName)
    if (!Files.exists(archivePath)) {
      archives.foreach(archive => {
        val currentPath = path.resolve(s"$archive.tgz")
        Manager.maybeDownload(currentPath, s"${EuroparlV7Manager.url}/$archive.tgz", bufferSize)
        CompressedFiles.decompressTGZ(currentPath, archivePath)
      })
    }
  }

//  def preprocess(
//      path: Path,
//      datasetType: DatasetType,
//      bufferSize: Int = 8192,
//      useMoses: Boolean = true,
//      numThreads: Int = 8
//  ): Unit = {
//    // Determine the appropriate tokenizer command
//    val tokenizeCommand = {
//      if (useMoses)
//        Seq(mosesDecoder.tokenizerScript.toAbsolutePath.toString, "-q", "-threads", numThreads.toString)
//      else
//        Seq(path.resolve("tools").resolve("tools").resolve("tokenizer.perl").toAbsolutePath.toString, "-q")
//    }
//
//    // Resolve paths
//    val dataPath = path.resolve(datasetType.name)
//    val srcTokenized = dataPath.resolve(datasetType.srcTokenizedCorpus)
//    val tgtTokenized = dataPath.resolve(datasetType.tgtTokenizedCorpus)
//    val srcTextFile = dataPath.resolve(datasetType.srcCorpus)
//    val tgtTextFile = dataPath.resolve(datasetType.tgtCorpus)
//
//    ((tokenizeCommand ++ Seq("-l", datasetType.srcLanguage)) #< srcTextFile.toFile #> srcTokenized.toFile).!
//    ((tokenizeCommand ++ Seq("-l", datasetType.tgtLanguage)) #< tgtTextFile.toFile #> tgtTokenized.toFile).!
//
//    val srcTextFiles = Set(srcTokenized)
//    val tgtTextFiles = Set(tgtTokenized)
//    Utilities.createVocab(srcTextFiles, dataPath.resolve(datasetType.srcVocab), bufferSize)
//    Utilities.createVocab(tgtTextFiles, dataPath.resolve(datasetType.tgtVocab), bufferSize)
//  }
}

object EuroparlV7Manager {
  private[EuroparlV7Manager] val logger = Logger(LoggerFactory.getLogger("Europarl v7 Data Manager"))

  val url: String = "http://www.statmt.org/europarl/v7"

  val supportedLanguagePairs: Set[(Language, Language)] = Set(
    (Bulgarian, English), (Czech, English), (Danish, English), (Dutch, English), (Estonian, English),
    (Finnish, English), (French, English), (German, English), (Greek, English), (Hungarian, English),
    (Italian, English), (Lithuanian, English), (Latvian, English), (Polish, English), (Portuguese, English),
    (Romanian, English), (Slovak, English), (Slovenian, English), (Spanish, English), (Swedish, English))

  def isLanguagePairSupported(srcLanguage: Language, tgtLanguage: Language): Boolean = {
    supportedLanguagePairs.contains((srcLanguage, tgtLanguage)) ||
        supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }
}
