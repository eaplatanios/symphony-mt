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

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.Language._

import better.files._
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.Path

/**
  * @author Emmanouil Antonios Platanios
  */
class NewsCommentaryV11Dataset(
    val srcLanguage: Language,
    val tgtLanguage: Language,
    override val workingDir: Path,
    override val bufferSize: Int = 8192,
    override val tokenize: Boolean = false
) extends Dataset(
  workingDir = workingDir
      .resolve("news-commentary-v11")
      .resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"),
  bufferSize = bufferSize,
  tokenize = tokenize
)(
  downloadsDir = workingDir.resolve("news-commentary-v11").resolve("downloads")
) {
  require(
    NewsCommentaryV11Dataset.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the News Commentary v11 dataset.")

  private[this] val src: String = srcLanguage.abbreviation
  private[this] val tgt: String = tgtLanguage.abbreviation

  private[this] val reversed: Boolean = {
    NewsCommentaryV11Dataset.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  private[this] val corpusFilenamePrefix: String = {
    s"news-commentary-v11.${if (reversed) s"$tgt-$src" else s"$src-$tgt"}"
  }

  /** Sequence of files to download as part of this dataset. */
  override val filesToDownload: Seq[String] = Seq(
    s"${NewsCommentaryV11Dataset.url}/${NewsCommentaryV11Dataset.archivePrefix}.tgz")

  /** Grouped files included in this dataset. */
  override val groupedFiles: Dataset.GroupedFiles = Dataset.GroupedFiles(
    trainCorpora = Seq(("NewsCommentaryV11/Train",
        File(super.workingDir) / NewsCommentaryV11Dataset.archivePrefix / s"$corpusFilenamePrefix.$src",
        File(super.workingDir) / NewsCommentaryV11Dataset.archivePrefix / s"$corpusFilenamePrefix.$tgt")))
}

object NewsCommentaryV11Dataset {
  private[NewsCommentaryV11Dataset] val logger = Logger(LoggerFactory.getLogger("News Commentary v11 Dataset"))

  val url          : String = "http://data.statmt.org/wmt16/translation-task"
  val archivePrefix: String = "training-parallel-nc-v11"

  val supportedLanguagePairs: Set[(Language, Language)] = {
    Set((Czech, English), (German, English), (Russian, English))
  }

  def isLanguagePairSupported(srcLanguage: Language, tgtLanguage: Language): Boolean = {
    supportedLanguagePairs.contains((srcLanguage, tgtLanguage)) ||
        supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }
}
