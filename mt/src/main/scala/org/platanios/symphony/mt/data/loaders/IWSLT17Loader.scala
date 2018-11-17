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
class IWSLT17Loader(
    override val srcLanguage: Language,
    override val tgtLanguage: Language,
    val config: DataConfig
) extends ParallelDatasetLoader(srcLanguage, tgtLanguage) {
  require(
    IWSLT17Loader.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the IWSLT-17 dataset.")

  override def name: String = "IWSLT-17"

  override def dataConfig: DataConfig = {
    config.copy(dataDir =
        config.dataDir
            .resolve("iwslt-17")
            .resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"))
  }

  override def downloadsDir: Path = config.dataDir.resolve("iwslt-17").resolve("downloads")

  /** Sequence of files to download as part of this dataset. */
  override def filesToDownload: Seq[(String, String)] = {
    Seq(
      (s"${IWSLT17Loader.url}/${IWSLT17Loader.filename}.tgz", s"${IWSLT17Loader.filename}.tgz"),
      (s"${IWSLT17Loader.testUrl}/$src/$tgt/$src-$tgt.tgz", s"$src-$tgt.tgz"),
      (s"${IWSLT17Loader.testUrl}/$tgt/$src/$tgt-$src.tgz", s"$tgt-$src.tgz"))
  }

  /** Returns all the corpora (tuples containing tag, source file, target file, and a file processor to use)
    * of this dataset type. */
  override def corpora(datasetType: DatasetType): Seq[(ParallelDataset.Tag, File, File, FileProcessor)] = {
    datasetType match {
      case Train => Seq((IWSLT17Loader.Train,
          File(downloadsDir) / IWSLT17Loader.filename / IWSLT17Loader.filename / s"train.tags.$src-$tgt.$src",
          File(downloadsDir) / IWSLT17Loader.filename / IWSLT17Loader.filename / s"train.tags.$src-$tgt.$tgt",
          TEDConverter >> Normalizer))
      case Dev => Seq((IWSLT17Loader.Dev2010,
          File(downloadsDir) / IWSLT17Loader.filename / IWSLT17Loader.filename / s"IWSLT17.TED.dev2010.$src-$tgt.$src.xml",
          File(downloadsDir) / IWSLT17Loader.filename / IWSLT17Loader.filename / s"IWSLT17.TED.dev2010.$src-$tgt.$tgt.xml",
          SGMConverter >> Normalizer))
      case Test => Seq(
        (IWSLT17Loader.Test2010,
            File(downloadsDir) / IWSLT17Loader.filename / IWSLT17Loader.filename / s"IWSLT17.TED.tst2010.$src-$tgt.$src.xml",
            File(downloadsDir) / IWSLT17Loader.filename / IWSLT17Loader.filename / s"IWSLT17.TED.tst2010.$src-$tgt.$tgt.xml",
            SGMConverter >> Normalizer),
        (IWSLT17Loader.Test2017,
            File(downloadsDir) / s"$src-$tgt" / s"$src-$tgt" / s"IWSLT17.TED.tst2017.mltlng.$src-$tgt.$src.xml",
            File(downloadsDir) / s"$tgt-$src" / s"$tgt-$src" / s"IWSLT17.TED.tst2017.mltlng.$tgt-$src.$tgt.xml",
            SGMConverter >> Normalizer))
    }
  }
}

object IWSLT17Loader {
  val url     : String = "https://wit3.fbk.eu/archive/2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo"
  val testUrl : String = "https://wit3.fbk.eu/archive/2017-01-mted-test/texts"
  val filename: String = "DeEnItNlRo-DeEnItNlRo"

  val supportedLanguagePairs: Set[(Language, Language)] = Set(
    (German, English), (German, Italian), (German, Dutch), (German, Romanian),
    (English, German), (English, Italian), (English, Dutch), (English, Romanian),
    (Italian, German), (Italian, English), (Italian, Dutch), (Italian, Romanian),
    (Dutch, German), (Dutch, Italian), (Dutch, English), (Dutch, Romanian),
    (Romanian, German), (Romanian, Italian), (Romanian, Dutch), (Romanian, English))

  def isLanguagePairSupported(srcLanguage: Language, tgtLanguage: Language): Boolean = {
    supportedLanguagePairs.contains((srcLanguage, tgtLanguage)) ||
        supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  def apply(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataConfig: DataConfig
  ): IWSLT17Loader = {
    new IWSLT17Loader(srcLanguage, tgtLanguage, dataConfig)
  }

  trait Tag extends ParallelDataset.Tag

  object Tag {
    @throws[IllegalArgumentException]
    def fromName(name: String): Tag = name match {
      case "train" => Train
      case "dev2010" => Dev2010
      case "tst2010" => Test2010
      case "tst2017" => Test2017
      case _ => throw new IllegalArgumentException(s"'$name' is not a valid IWSLT-17 tag.")
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

  case object Test2017 extends Tag {
    override val value: String = "tst2017"
  }
}
