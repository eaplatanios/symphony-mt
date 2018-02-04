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
class CommonCrawlDataset(
    override protected val workingDir: Path,
    override val srcLanguage: Language,
    override val tgtLanguage: Language,
    override val bufferSize: Int = 8192,
    override val tokenize: Boolean = false,
    override val trainDataSentenceLengthBounds: (Int, Int) = null
) extends Dataset(
  workingDir = workingDir.resolve("commoncrawl").resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"),
  srcLanguage = srcLanguage,
  tgtLanguage = tgtLanguage,
  bufferSize = bufferSize,
  tokenize = tokenize,
  trainDataSentenceLengthBounds = trainDataSentenceLengthBounds
)(
  downloadsDir = workingDir.resolve("commoncrawl")
) {
  require(
    CommonCrawlDataset.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the CommonCrawl dataset.")

  override def name: String = "CommonCrawl"

  private[this] def reversed: Boolean = {
    CommonCrawlDataset.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  private[this] def corpusFilenamePrefix: String = {
    s"commoncrawl.${if (reversed) s"$tgt-$src" else s"$src-$tgt"}"
  }

  /** Sequence of files to download as part of this dataset. */
  override def filesToDownload: Seq[String] = Seq(
    s"${CommonCrawlDataset.url}/${CommonCrawlDataset.archivePrefix}.tgz")

  /** Grouped files included in this dataset. */
  override private[data] def groupFiles: Dataset.GroupedFiles = Dataset.GroupedFiles(
    trainCorpora = Seq(("CommonCrawl/Train",
        File(downloadsDir) / CommonCrawlDataset.archivePrefix / s"$corpusFilenamePrefix.$src",
        File(downloadsDir) / CommonCrawlDataset.archivePrefix / s"$corpusFilenamePrefix.$tgt")))
}

object CommonCrawlDataset {
  val url          : String = "http://www.statmt.org/wmt13"
  val archivePrefix: String = "training-parallel-commoncrawl"

  val supportedLanguagePairs: Set[(Language, Language)] = Set(
    (Czech, English), (French, English), (German, English), (Russian, English), (Spanish, English))

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
  ): CommonCrawlDataset = {
    new CommonCrawlDataset(workingDir, srcLanguage, tgtLanguage, bufferSize, tokenize, trainDataSentenceLengthBounds)
  }
}
