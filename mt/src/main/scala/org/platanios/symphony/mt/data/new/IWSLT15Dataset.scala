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
import org.platanios.symphony.mt.Language.{English, Vietnamese}

import better.files._
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.Path

/**
  * @author Emmanouil Antonios Platanios
  */
class IWSLT15Dataset(
    val srcLanguage: Language,
    val tgtLanguage: Language,
    override protected val workingDir: Path,
    override val bufferSize: Int = 8192
) extends Dataset(
  workingDir = workingDir.resolve("iwslt-15").resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"),
  bufferSize = bufferSize,
  tokenize = false
)(
  downloadsDir = workingDir.resolve("iwslt-15").resolve("downloads")
) {
  require(
    IWSLT15Dataset.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the IWSLT-15 dataset.")

  override def dataDir: Path = {
    workingDir.resolve("iwslt-15").resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}")
  }

  private[this] val src: String = srcLanguage.abbreviation
  private[this] val tgt: String = tgtLanguage.abbreviation

  private[this] val reversed: Boolean = {
    IWSLT15Dataset.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  private[this] val directoryName: String = if (reversed) s"iwslt15.$tgt-$src" else s"iwslt15.$src-$tgt"

  /** Sequence of files to download as part of this dataset. */
  override def filesToDownload: Seq[String] = Seq(
    s"${IWSLT15Dataset.url}/$directoryName/${IWSLT15Dataset.trainPrefix}.$src",
    s"${IWSLT15Dataset.url}/$directoryName/${IWSLT15Dataset.trainPrefix}.$tgt",
    s"${IWSLT15Dataset.url}/$directoryName/${IWSLT15Dataset.devPrefix}.$src",
    s"${IWSLT15Dataset.url}/$directoryName/${IWSLT15Dataset.devPrefix}.$tgt",
    s"${IWSLT15Dataset.url}/$directoryName/${IWSLT15Dataset.testPrefix}.$src",
    s"${IWSLT15Dataset.url}/$directoryName/${IWSLT15Dataset.testPrefix}.$tgt",
    s"${IWSLT15Dataset.url}/$directoryName/${IWSLT15Dataset.vocabPrefix}.$src",
    s"${IWSLT15Dataset.url}/$directoryName/${IWSLT15Dataset.vocabPrefix}.$tgt")

  /** Grouped files included in this dataset. */
  override def groupedFiles: Dataset.GroupedFiles = Dataset.GroupedFiles(
    trainCorpora = Seq(("IWSLT15/Train",
        File(dataDir) / s"${IWSLT15Dataset.trainPrefix}.$src",
        File(dataDir) / s"${IWSLT15Dataset.trainPrefix}.$tgt")),
    devCorpora = Seq(("IWSLT15/Dev",
        File(dataDir) / s"${IWSLT15Dataset.devPrefix}.$src",
        File(dataDir) / s"${IWSLT15Dataset.devPrefix}.$tgt")),
    testCorpora = Seq(("IWSLT15/Test",
        File(dataDir) / s"${IWSLT15Dataset.testPrefix}.$src",
        File(dataDir) / s"${IWSLT15Dataset.testPrefix}.$tgt")),
    vocabularies = Some((
        File(dataDir) / s"${IWSLT15Dataset.vocabPrefix}.$src",
        File(dataDir) / s"${IWSLT15Dataset.vocabPrefix}.$tgt")))
}

object IWSLT15Dataset {
  private[IWSLT15Dataset] val logger = Logger(LoggerFactory.getLogger("IWSLT-15 Dataset"))

  val url        : String = "https://nlp.stanford.edu/projects/nmt/data"
  val trainPrefix: String = "train"
  val devPrefix  : String = "tst2012"
  val testPrefix : String = "tst2013"
  val vocabPrefix: String = "vocab"

  val supportedLanguagePairs: Set[(Language, Language)] = Set((Vietnamese, English))

  def isLanguagePairSupported(srcLanguage: Language, tgtLanguage: Language): Boolean = {
    supportedLanguagePairs.contains((srcLanguage, tgtLanguage)) ||
        supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  def apply(
      srcLanguage: Language,
      tgtLanguage: Language,
      workingDir: Path,
      bufferSize: Int = 8192
  ): IWSLT15Dataset = {
    new IWSLT15Dataset(srcLanguage, tgtLanguage, workingDir, bufferSize)
  }
}
