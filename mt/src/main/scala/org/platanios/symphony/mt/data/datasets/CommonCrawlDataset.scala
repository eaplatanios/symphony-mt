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

package org.platanios.symphony.mt.data.datasets

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.Language._
import org.platanios.symphony.mt.data.{DataConfig, Dataset}

import better.files._

import java.nio.file.Path

/**
  * @author Emmanouil Antonios Platanios
  */
class CommonCrawlDataset(
    override val srcLanguage: Language,
    override val tgtLanguage: Language,
    val config: DataConfig
) extends Dataset(srcLanguage = srcLanguage, tgtLanguage = tgtLanguage) {
  require(
    CommonCrawlDataset.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the CommonCrawl dataset.")

  override def name: String = "CommonCrawl"

  override def dataConfig: DataConfig = {
    config.copy(workingDir =
        config.workingDir
            .resolve("commoncrawl")
            .resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"))
  }

  override def downloadsDir: Path = config.workingDir.resolve("commoncrawl").resolve("downloads")

  private[this] def reversed: Boolean = {
    CommonCrawlDataset.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  private[this] def corpusFilenamePrefix: String = {
    s"commoncrawl.${if (reversed) s"$tgt-$src" else s"$src-$tgt"}"
  }

  /** Sequence of files to download as part of this dataset. */
  override def filesToDownload: Seq[String] = Seq(
    s"${CommonCrawlDataset.url}/${CommonCrawlDataset.archivePrefix}.tgz")

  /** Returns all the train corpora (tuples containing name, source file, and target file) of this dataset. */
  override def trainCorpora: Seq[(String, File, File)] = Seq(("CommonCrawl/Train",
      File(downloadsDir) / CommonCrawlDataset.archivePrefix / s"$corpusFilenamePrefix.$src",
      File(downloadsDir) / CommonCrawlDataset.archivePrefix / s"$corpusFilenamePrefix.$tgt"))
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
      srcLanguage: Language,
      tgtLanguage: Language,
      dataConfig: DataConfig
  ): CommonCrawlDataset = {
    new CommonCrawlDataset(srcLanguage, tgtLanguage, dataConfig)
  }
}
