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
import org.platanios.symphony.mt.Language.{English, Vietnamese}
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.data.processors.{FileProcessor, NoFileProcessor}

import better.files._

import java.nio.file.Path

/**
  * @author Emmanouil Antonios Platanios
  */
class IWSLT15StanfordLoader(
    override val srcLanguage: Language,
    override val tgtLanguage: Language,
    val config: DataConfig
) extends ParallelDatasetLoader(srcLanguage, tgtLanguage) {
  require(
    IWSLT15StanfordLoader.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the IWSLT-15 dataset.")

  override def name: String = "IWSLT-15"

  override def dataConfig: DataConfig = {
    config.copy(dataDir =
        config.dataDir
            .resolve("iwslt-15")
            .resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"))
  }

  override def downloadsDir: Path = config.dataDir.resolve("iwslt-15").resolve("downloads")

  private[this] def reversed: Boolean = {
    IWSLT15StanfordLoader.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  private[this] def directoryName: String = if (reversed) s"iwslt15.$tgt-$src" else s"iwslt15.$src-$tgt"

  /** Sequence of files to download as part of this dataset. */
  override def filesToDownload: Seq[(String, String)] = {
    Seq(
      s"${IWSLT15StanfordLoader.url}/$directoryName/${IWSLT15StanfordLoader.trainPrefix}.$src",
      s"${IWSLT15StanfordLoader.url}/$directoryName/${IWSLT15StanfordLoader.trainPrefix}.$tgt",
      s"${IWSLT15StanfordLoader.url}/$directoryName/${IWSLT15StanfordLoader.devPrefix}.$src",
      s"${IWSLT15StanfordLoader.url}/$directoryName/${IWSLT15StanfordLoader.devPrefix}.$tgt",
      s"${IWSLT15StanfordLoader.url}/$directoryName/${IWSLT15StanfordLoader.testPrefix}.$src",
      s"${IWSLT15StanfordLoader.url}/$directoryName/${IWSLT15StanfordLoader.testPrefix}.$tgt",
      s"${IWSLT15StanfordLoader.url}/$directoryName/${IWSLT15StanfordLoader.vocabPrefix}.$src",
      s"${IWSLT15StanfordLoader.url}/$directoryName/${IWSLT15StanfordLoader.vocabPrefix}.$tgt"
    ).map(url => (url, url.splitAt(url.lastIndexOf('/') + 1)._2))
  }

  /** Returns all the corpora (tuples containing tag, source file, target file, and a file processor to use)
    * of this dataset type. */
  override def corpora(datasetType: DatasetType): Seq[(ParallelDataset.Tag, File, File, FileProcessor)] = {
    datasetType match {
      case Train => Seq((IWSLT15StanfordLoader.Train,
          File(downloadsDir) / s"${IWSLT15StanfordLoader.trainPrefix}.$src",
          File(downloadsDir) / s"${IWSLT15StanfordLoader.trainPrefix}.$tgt", NoFileProcessor))
      case Dev => Seq((IWSLT15StanfordLoader.Test2012,
          File(downloadsDir) / s"${IWSLT15StanfordLoader.devPrefix}.$src",
          File(downloadsDir) / s"${IWSLT15StanfordLoader.devPrefix}.$tgt", NoFileProcessor))
      case Test => Seq((IWSLT15StanfordLoader.Test2013,
          File(downloadsDir) / s"${IWSLT15StanfordLoader.testPrefix}.$src",
          File(downloadsDir) / s"${IWSLT15StanfordLoader.testPrefix}.$tgt", NoFileProcessor))
    }
  }

  /** Returns the source and the target vocabulary of this dataset. */
  override def vocabularies: (Seq[File], Seq[File]) = (
      Seq(File(downloadsDir) / s"${IWSLT15StanfordLoader.vocabPrefix}.$src"),
      Seq(File(downloadsDir) / s"${IWSLT15StanfordLoader.vocabPrefix}.$tgt"))
}

object IWSLT15StanfordLoader {
  val url        : String = "https://nlp.stanford.edu/projects/nmt/data"
  val trainPrefix: String = "train"
  val devPrefix  : String = "tst2012"
  val testPrefix : String = "tst2013"
  val vocabPrefix: String = "vocab"

  val supportedLanguagePairs: Set[(Language, Language)] = Set((English, Vietnamese))

  def isLanguagePairSupported(srcLanguage: Language, tgtLanguage: Language): Boolean = {
    supportedLanguagePairs.contains((srcLanguage, tgtLanguage)) ||
        supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  def apply(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataConfig: DataConfig
  ): IWSLT15StanfordLoader = {
    new IWSLT15StanfordLoader(srcLanguage, tgtLanguage, dataConfig)
  }

  trait Tag extends ParallelDataset.Tag

  object Tag {
    @throws[IllegalArgumentException]
    def fromName(name: String): Tag = name match {
      case "train" => Train
      case "tst2012" => Test2012
      case "tst2013" => Test2013
      case _ => throw new IllegalArgumentException(s"'$name' is not a valid IWSLT-15 Stanford tag.")
    }
  }

  case object Train extends Tag {
    override val value: String = "train"
  }

  case object Test2012 extends Tag {
    override val value: String = "tst2012"
  }

  case object Test2013 extends Tag {
    override val value: String = "tst2013"
  }
}
