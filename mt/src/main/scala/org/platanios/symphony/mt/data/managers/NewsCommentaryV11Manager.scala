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
case class NewsCommentaryV11Manager(srcLanguage: Language, tgtLanguage: Language) extends Manager {
  require(
    isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the News Commentary v11 data manager.")

  val src: String = srcLanguage.abbreviation
  val tgt: String = tgtLanguage.abbreviation

  val name: String = s"$src-$tgt"

  override val supportedLanguagePairs: Set[(Language, Language)] = {
    Set((Czech, English), (German, English), (Russian, English))
  }

  private[this] val reversed: Boolean = supportedLanguagePairs.contains((tgtLanguage, srcLanguage))

  private[this] val corpusFilenamePrefix: String = {
    s"news-commentary-v11.${if (reversed) s"$tgt-$src" else s"$src-$tgt"}"
  }

  def download(path: Path, bufferSize: Int = 8192): ParallelDataset = {
    val processedPath = path.resolve("nc-v11")

    // Download and decompress the data, if necessary.
    val archivePathPrefix = processedPath.resolve(s"${NewsCommentaryV11Manager.archivePrefix}")
    val archivePath = processedPath.resolve(s"${NewsCommentaryV11Manager.archivePrefix}.tgz")
    val srcTrainCorpus = archivePathPrefix.resolve(s"$corpusFilenamePrefix.$src")
    val tgtTrainCorpus = archivePathPrefix.resolve(s"$corpusFilenamePrefix.$tgt")

    if (!Files.exists(archivePathPrefix)) {
      Manager.maybeDownload(
        archivePath, s"${NewsCommentaryV11Manager.url}/${NewsCommentaryV11Manager.archivePrefix}.tgz", bufferSize)
      CompressedFiles.decompressTGZ(archivePath, archivePathPrefix, bufferSize)
    }

    ParallelDataset(trainCorpora = Map(srcLanguage -> Seq(srcTrainCorpus), tgtLanguage -> Seq(tgtTrainCorpus)))
  }
}

object NewsCommentaryV11Manager {
  private[NewsCommentaryV11Manager] val logger = Logger(LoggerFactory.getLogger("News Commentary v11 Data Manager"))

  val url          : String = "http://data.statmt.org/wmt16/translation-task"
  val archivePrefix: String = "training-parallel-nc-v11"
}
