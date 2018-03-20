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
import org.platanios.symphony.mt.data.processors.{FileProcessor, NoFileProcessor}

import better.files._

import java.nio.file.Path

/**
  * @author Emmanouil Antonios Platanios
  */
class IWSLT15StanfordDatasetLoader(
    override val srcLanguage: Language,
    override val tgtLanguage: Language,
    val config: DataConfig
) extends ParallelDatasetLoader(srcLanguage, tgtLanguage) {
  require(
    IWSLT15StanfordDatasetLoader.isLanguagePairSupported(srcLanguage, tgtLanguage),
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
    IWSLT15StanfordDatasetLoader.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  private[this] def directoryName: String = if (reversed) s"iwslt15.$tgt-$src" else s"iwslt15.$src-$tgt"

  /** Sequence of files to download as part of this dataset. */
  override def filesToDownload: Seq[String] = Seq(
    s"${IWSLT15StanfordDatasetLoader.url}/$directoryName/${IWSLT15StanfordDatasetLoader.trainPrefix}.$src",
    s"${IWSLT15StanfordDatasetLoader.url}/$directoryName/${IWSLT15StanfordDatasetLoader.trainPrefix}.$tgt",
    s"${IWSLT15StanfordDatasetLoader.url}/$directoryName/${IWSLT15StanfordDatasetLoader.devPrefix}.$src",
    s"${IWSLT15StanfordDatasetLoader.url}/$directoryName/${IWSLT15StanfordDatasetLoader.devPrefix}.$tgt",
    s"${IWSLT15StanfordDatasetLoader.url}/$directoryName/${IWSLT15StanfordDatasetLoader.testPrefix}.$src",
    s"${IWSLT15StanfordDatasetLoader.url}/$directoryName/${IWSLT15StanfordDatasetLoader.testPrefix}.$tgt",
    s"${IWSLT15StanfordDatasetLoader.url}/$directoryName/${IWSLT15StanfordDatasetLoader.vocabPrefix}.$src",
    s"${IWSLT15StanfordDatasetLoader.url}/$directoryName/${IWSLT15StanfordDatasetLoader.vocabPrefix}.$tgt")

  /** Returns all the corpora (tuples containing name, source file, target file, and a file processor to use)
    * of this dataset type. */
  override def corpora(datasetType: DatasetType): Seq[(String, File, File, FileProcessor)] = datasetType match {
    case Train => Seq(("Train",
        File(downloadsDir) / s"${IWSLT15StanfordDatasetLoader.trainPrefix}.$src",
        File(downloadsDir) / s"${IWSLT15StanfordDatasetLoader.trainPrefix}.$tgt", NoFileProcessor))
    case Dev => Seq(("Dev",
        File(downloadsDir) / s"${IWSLT15StanfordDatasetLoader.devPrefix}.$src",
        File(downloadsDir) / s"${IWSLT15StanfordDatasetLoader.devPrefix}.$tgt", NoFileProcessor))
    case Test => Seq(("Test",
        File(downloadsDir) / s"${IWSLT15StanfordDatasetLoader.testPrefix}.$src",
        File(downloadsDir) / s"${IWSLT15StanfordDatasetLoader.testPrefix}.$tgt", NoFileProcessor))
  }

  /** Returns the source and the target vocabulary of this dataset. */
  override def vocabularies: (Seq[File], Seq[File]) = (
      Seq(File(downloadsDir) / s"${IWSLT15StanfordDatasetLoader.vocabPrefix}.$src"),
      Seq(File(downloadsDir) / s"${IWSLT15StanfordDatasetLoader.vocabPrefix}.$tgt"))
}

object IWSLT15StanfordDatasetLoader {
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
  ): IWSLT15StanfordDatasetLoader = {
    new IWSLT15StanfordDatasetLoader(srcLanguage, tgtLanguage, dataConfig)
  }
}
