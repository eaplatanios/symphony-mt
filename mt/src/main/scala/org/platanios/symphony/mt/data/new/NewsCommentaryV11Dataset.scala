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

import java.nio.file.Path

/**
  * @author Emmanouil Antonios Platanios
  */
class NewsCommentaryV11Dataset(
    override protected val workingDir: Path,
    override val srcLanguage: Language,
    override val tgtLanguage: Language,
    override val bufferSize: Int = 8192,
    override val tokenize: Boolean = false,
    override val trainDataSentenceLengthBounds: (Int, Int) = null
) extends Dataset(
  workingDir = workingDir
      .resolve("news-commentary-v11")
      .resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"),
  srcLanguage = srcLanguage,
  tgtLanguage = tgtLanguage,
  bufferSize = bufferSize,
  tokenize = tokenize,
  trainDataSentenceLengthBounds = trainDataSentenceLengthBounds
)(
  downloadsDir = workingDir.resolve("news-commentary-v11")
) {
  require(
    NewsCommentaryV11Dataset.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the News Commentary v11 dataset.")

  override def name: String = "News Commentary v11"

  private[this] def reversed: Boolean = {
    NewsCommentaryV11Dataset.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  private[this] def corpusFilenamePrefix: String = {
    s"news-commentary-v11.${if (reversed) s"$tgt-$src" else s"$src-$tgt"}"
  }

  /** Sequence of files to download as part of this dataset. */
  override def filesToDownload: Seq[String] = Seq(
    s"${NewsCommentaryV11Dataset.url}/${NewsCommentaryV11Dataset.archivePrefix}.tgz")

  /** Grouped files included in this dataset. */
  override private[data] def groupFiles: Dataset.GroupedFiles = Dataset.GroupedFiles(
    trainCorpora = Seq(("NewsCommentaryV11/Train",
        File(downloadsDir) / NewsCommentaryV11Dataset.archivePrefix
            / NewsCommentaryV11Dataset.archivePrefix / s"$corpusFilenamePrefix.$src",
        File(downloadsDir) / NewsCommentaryV11Dataset.archivePrefix
            / NewsCommentaryV11Dataset.archivePrefix / s"$corpusFilenamePrefix.$tgt")))
}

object NewsCommentaryV11Dataset {
  val url          : String = "http://data.statmt.org/wmt16/translation-task"
  val archivePrefix: String = "training-parallel-nc-v11"

  val supportedLanguagePairs: Set[(Language, Language)] = {
    Set((Czech, English), (German, English), (Russian, English))
  }

  def isLanguagePairSupported(srcLanguage: Language, tgtLanguage: Language): Boolean = {
    supportedLanguagePairs.contains((srcLanguage, tgtLanguage)) ||
        supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  def apply(
      workingDir: Path,
      srcLanguage: Language,
      tgtLanguage: Language,
      bufferSize: Int = 8192,
      tokenize: Boolean = false,
      trainDataSentenceLengthBounds: (Int, Int) = null
  ): NewsCommentaryV11Dataset = {
    new NewsCommentaryV11Dataset(
      workingDir, srcLanguage, tgtLanguage, bufferSize, tokenize, trainDataSentenceLengthBounds)
  }
}
