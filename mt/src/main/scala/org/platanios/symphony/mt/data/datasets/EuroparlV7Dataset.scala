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
class EuroparlV7Dataset(
    override val srcLanguage: Language,
    override val tgtLanguage: Language,
    val config: DataConfig
) extends Dataset(srcLanguage = srcLanguage, tgtLanguage = tgtLanguage) {
  require(
    EuroparlV7Dataset.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the Europarl v7 dataset.")

  override def name: String = "Europarl v7"

  override def dataConfig: DataConfig = {
    config.copy(workingDir =
        config.workingDir
            .resolve("europarl-v7")
            .resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"))
  }

  override def downloadsDir: Path = config.workingDir.resolve("europarl-v7").resolve("downloads")

  private[this] def reversed: Boolean = {
    EuroparlV7Dataset.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  private[this] def corpusArchiveFile: String = if (reversed) s"$tgt-$src" else s"$src-$tgt"

  private[this] def corpusFilenamePrefix: String = {
    s"europarl-v7.${if (reversed) s"$tgt-$src" else s"$src-$tgt"}"
  }

  /** Sequence of files to download as part of this dataset. */
  override def filesToDownload: Seq[String] = Seq(
    s"${EuroparlV7Dataset.url}/$corpusArchiveFile.tgz")

  /** Returns all the train corpora (tuples containing name, source file, and target file) of this dataset. */
  override def trainCorpora: Seq[(String, File, File)] = Seq(("EuroparlV7/Train",
      File(downloadsDir) / corpusArchiveFile / s"$corpusFilenamePrefix.$src",
      File(downloadsDir) / corpusArchiveFile / s"$corpusFilenamePrefix.$tgt"))
}

object EuroparlV7Dataset {
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

  def apply(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataConfig: DataConfig
  ): EuroparlV7Dataset = {
    new EuroparlV7Dataset(srcLanguage, tgtLanguage, dataConfig)
  }
}
