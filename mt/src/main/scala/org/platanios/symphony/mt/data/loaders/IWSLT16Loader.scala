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
import org.platanios.symphony.mt.Language._
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.data.processors._

import better.files._

import java.nio.file.Path

/**
  * @author Emmanouil Antonios Platanios
  */
class IWSLT16Loader(
    override val srcLanguage: Language,
    override val tgtLanguage: Language,
    val config: DataConfig
) extends ParallelDatasetLoader(srcLanguage, tgtLanguage) {
  require(
    IWSLT16Loader.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the IWSLT-16 dataset.")

  override def name: String = "IWSLT-16"

  override def dataConfig: DataConfig = {
    config.copy(dataDir =
        config.dataDir
            .resolve("iwslt-16")
            .resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"))
  }

  override def downloadsDir: Path = config.dataDir.resolve("iwslt-16").resolve("downloads")

  private[this] def directoryName: String = s"$src-$tgt"

  /** Sequence of files to download as part of this dataset. */
  override def filesToDownload: Seq[String] = Seq(s"${IWSLT16Loader.url}/$src/$tgt/$directoryName.tgz")

  /** Returns all the corpora (tuples containing tag, source file, target file, and a file processor to use)
    * of this dataset type. */
  override def corpora(datasetType: DatasetType): Seq[(ParallelDataset.Tag, File, File, FileProcessor)] = {
    datasetType match {
      case Train => Seq((IWSLT16Loader.Train,
          File(downloadsDir) / directoryName / directoryName / s"train.tags.$directoryName.$src",
          File(downloadsDir) / directoryName / directoryName / s"train.tags.$directoryName.$tgt",
          TEDConverter >> Normalizer))
      case Dev => Seq((IWSLT16Loader.Dev2010,
          File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.dev2010.$directoryName.$src.xml",
          File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.dev2010.$directoryName.$tgt.xml",
          SGMConverter >> Normalizer))
      case Test => Seq(
        (IWSLT16Loader.Test2010,
            File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2010.$directoryName.$src.xml",
            File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2010.$directoryName.$tgt.xml",
            SGMConverter >> Normalizer),
        (IWSLT16Loader.Test2011,
            File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2011.$directoryName.$src.xml",
            File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2011.$directoryName.$tgt.xml",
            SGMConverter >> Normalizer),
        (IWSLT16Loader.Test2012,
            File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2012.$directoryName.$src.xml",
            File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2012.$directoryName.$tgt.xml",
            SGMConverter >> Normalizer),
        (IWSLT16Loader.Test2013,
            File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2013.$directoryName.$src.xml",
            File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2013.$directoryName.$tgt.xml",
            SGMConverter >> Normalizer),
        (IWSLT16Loader.Test2014,
            File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2014.$directoryName.$src.xml",
            File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2014.$directoryName.$tgt.xml",
            SGMConverter >> Normalizer))
    }
  }
}

object IWSLT16Loader {
  val url: String = "https://wit3.fbk.eu/archive/2016-01/texts"

  val supportedLanguagePairs: Set[(Language, Language)] = Set(
    (English, Arabic), (English, Czech), (English, German), (English, French),
    (Arabic, English), (Czech, English), (German, English), (French, English))

  def isLanguagePairSupported(srcLanguage: Language, tgtLanguage: Language): Boolean = {
    supportedLanguagePairs.contains((srcLanguage, tgtLanguage)) ||
        supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  def apply(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataConfig: DataConfig
  ): IWSLT16Loader = {
    new IWSLT16Loader(srcLanguage, tgtLanguage, dataConfig)
  }

  trait Tag extends ParallelDataset.Tag

  object Tag {
    @throws[IllegalArgumentException]
    def fromName(name: String): Tag = name match {
      case "train" => Train
      case "dev2010" => Dev2010
      case "tst2010" => Test2010
      case "tst2011" => Test2011
      case "tst2012" => Test2012
      case "tst2013" => Test2013
      case "tst2014" => Test2014
      case _ => throw new IllegalArgumentException(s"'$name' is not a valid IWSLT-16 tag.")
    }
  }

  case object Train extends Tag {
    override val value: String = "train"
  }

  case object Dev2010 extends Tag {
    override val value: String = "dev2010"
  }

  case object Test2010 extends Tag {
    override val value: String = "tst2010"
  }

  case object Test2011 extends Tag {
    override val value: String = "tst2011"
  }

  case object Test2012 extends Tag {
    override val value: String = "tst2012"
  }

  case object Test2013 extends Tag {
    override val value: String = "tst2013"
  }

  case object Test2014 extends Tag {
    override val value: String = "tst2014"
  }
}
