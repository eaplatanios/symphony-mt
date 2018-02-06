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
import org.platanios.symphony.mt.data.Dataset

import better.files._

import java.nio.file.Path

/**
  * @author Emmanouil Antonios Platanios
  */
class EuroparlV8Dataset(
    protected val dataDir: Path,
    override val srcLanguage: Language,
    override val tgtLanguage: Language,
    override val bufferSize: Int = 8192,
    override val tokenize: Boolean = false,
    override val trainDataSentenceLengthBounds: (Int, Int) = null
) extends Dataset(
  workingDir = dataDir.resolve("europarl-v8").resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"),
  srcLanguage = srcLanguage,
  tgtLanguage = tgtLanguage,
  bufferSize = bufferSize,
  tokenize = tokenize,
  trainDataSentenceLengthBounds = trainDataSentenceLengthBounds
)(
  downloadsDir = dataDir.resolve("europarl-v8")
) {
  require(
    EuroparlV8Dataset.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the Europarl v8 dataset.")

  override def name: String = "Europarl v8"

  private[this] def reversed: Boolean = {
    EuroparlV8Dataset.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  private[this] def corpusFilenamePrefix: String = {
    s"europarl-v8.${if (reversed) s"$tgt-$src" else s"$src-$tgt"}"
  }

  /** Sequence of files to download as part of this dataset. */
  override def filesToDownload: Seq[String] = Seq(
    s"${EuroparlV8Dataset.url}/${EuroparlV8Dataset.archivePrefix}.tgz")

  /** Returns all the train corpora (tuples containing name, source file, and target file) of this dataset. */
  override def trainCorpora: Seq[(String, File, File)] = Seq(("EuroparlV8/Train",
      File(downloadsDir) / EuroparlV8Dataset.archivePrefix / s"$corpusFilenamePrefix.$src",
      File(downloadsDir) / EuroparlV8Dataset.archivePrefix / s"$corpusFilenamePrefix.$tgt"))
}

object EuroparlV8Dataset {
  val url          : String = "http://data.statmt.org/wmt16/translation-task"
  val archivePrefix: String = "training-parallel-ep-v8"

  val supportedLanguagePairs: Set[(Language, Language)] = Set((Finnish, English), (Romanian, English))

  def isLanguagePairSupported(srcLanguage: Language, tgtLanguage: Language): Boolean = {
    supportedLanguagePairs.contains((srcLanguage, tgtLanguage)) ||
        supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  def apply(
      dataDir: Path,
      srcLanguage: Language,
      tgtLanguage: Language,
      bufferSize: Int = 8192,
      tokenize: Boolean = false,
      trainDataSentenceLengthBounds: (Int, Int) = null
  ): EuroparlV8Dataset = {
    new EuroparlV8Dataset(dataDir, srcLanguage, tgtLanguage, bufferSize, tokenize, trainDataSentenceLengthBounds)
  }
}
