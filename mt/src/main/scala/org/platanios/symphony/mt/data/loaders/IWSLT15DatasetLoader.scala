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

package org.platanios.symphony.mt.data.loaders

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.Language.{english, vietnamese}
import org.platanios.symphony.mt.data._

import better.files._

import java.nio.file.Path

/**
  * @author Emmanouil Antonios Platanios
  */
class IWSLT15DatasetLoader(
    override val srcLanguage: Language,
    override val tgtLanguage: Language,
    val config: DataConfig
) extends ParallelDatasetLoader(srcLanguage = srcLanguage, tgtLanguage = tgtLanguage) {
  require(
    IWSLT15DatasetLoader.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the IWSLT-15 dataset.")

  override def name: String = "IWSLT-15"

  override def dataConfig: DataConfig = {
    config.copy(workingDir =
        config.workingDir
            .resolve("iwslt-15")
            .resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"))
  }

  override def downloadsDir: Path = config.workingDir.resolve("iwslt-15").resolve("downloads")

  private[this] def reversed: Boolean = {
    IWSLT15DatasetLoader.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  private[this] def directoryName: String = if (reversed) s"iwslt15.$tgt-$src" else s"iwslt15.$src-$tgt"

  /** Sequence of files to download as part of this dataset. */
  override def filesToDownload: Seq[String] = Seq(
    s"${IWSLT15DatasetLoader.url}/$directoryName/${IWSLT15DatasetLoader.trainPrefix}.$src",
    s"${IWSLT15DatasetLoader.url}/$directoryName/${IWSLT15DatasetLoader.trainPrefix}.$tgt",
    s"${IWSLT15DatasetLoader.url}/$directoryName/${IWSLT15DatasetLoader.devPrefix}.$src",
    s"${IWSLT15DatasetLoader.url}/$directoryName/${IWSLT15DatasetLoader.devPrefix}.$tgt",
    s"${IWSLT15DatasetLoader.url}/$directoryName/${IWSLT15DatasetLoader.testPrefix}.$src",
    s"${IWSLT15DatasetLoader.url}/$directoryName/${IWSLT15DatasetLoader.testPrefix}.$tgt",
    s"${IWSLT15DatasetLoader.url}/$directoryName/${IWSLT15DatasetLoader.vocabPrefix}.$src",
    s"${IWSLT15DatasetLoader.url}/$directoryName/${IWSLT15DatasetLoader.vocabPrefix}.$tgt")

  /** Returns all the corpora (tuples containing name, source file, and target file) of this dataset type. */
  override def corpora(datasetType: DatasetType): Seq[(String, File, File)] = datasetType match {
    case Train => Seq(("IWSLT15/Train",
        File(downloadsDir) / s"${IWSLT15DatasetLoader.trainPrefix}.$src",
        File(downloadsDir) / s"${IWSLT15DatasetLoader.trainPrefix}.$tgt"))
    case Dev => Seq(("IWSLT15/Dev",
        File(downloadsDir) / s"${IWSLT15DatasetLoader.devPrefix}.$src",
        File(downloadsDir) / s"${IWSLT15DatasetLoader.devPrefix}.$tgt"))
    case Test => Seq(("IWSLT15/Test",
        File(downloadsDir) / s"${IWSLT15DatasetLoader.testPrefix}.$src",
        File(downloadsDir) / s"${IWSLT15DatasetLoader.testPrefix}.$tgt"))
  }

  /** Returns the source and the target vocabulary of this dataset. */
  override def vocabularies: (Seq[File], Seq[File]) = (
      Seq(File(downloadsDir) / s"${IWSLT15DatasetLoader.vocabPrefix}.$src"),
      Seq(File(downloadsDir) / s"${IWSLT15DatasetLoader.vocabPrefix}.$tgt"))
}

object IWSLT15DatasetLoader {
  val url        : String = "https://nlp.stanford.edu/projects/nmt/data"
  val trainPrefix: String = "train"
  val devPrefix  : String = "tst2012"
  val testPrefix : String = "tst2013"
  val vocabPrefix: String = "vocab"

  val supportedLanguagePairs: Set[(Language, Language)] = Set((english, vietnamese))

  def isLanguagePairSupported(srcLanguage: Language, tgtLanguage: Language): Boolean = {
    supportedLanguagePairs.contains((srcLanguage, tgtLanguage)) ||
        supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  def apply(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataConfig: DataConfig
  ): IWSLT15DatasetLoader = {
    new IWSLT15DatasetLoader(srcLanguage, tgtLanguage, dataConfig)
  }
}
