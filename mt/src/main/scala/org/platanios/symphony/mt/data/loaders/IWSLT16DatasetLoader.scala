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
import org.platanios.symphony.mt.data.processors.{FileProcessor, SGMConverter, TEDConverter}

import better.files._

import java.nio.file.Path

/**
  * @author Emmanouil Antonios Platanios
  */
class IWSLT16DatasetLoader(
    override val srcLanguage: Language,
    override val tgtLanguage: Language,
    val config: DataConfig
) extends ParallelDatasetLoader(srcLanguage, tgtLanguage) {
  require(
    IWSLT16DatasetLoader.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the IWSLT-16 dataset.")

  override def name: String = "IWSLT-16"

  override def dataConfig: DataConfig = {
    config.copy(workingDir =
        config.workingDir
            .resolve("iwslt-16")
            .resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"))
  }

  override def downloadsDir: Path = config.workingDir.resolve("iwslt-16").resolve("downloads")

  private[this] def directoryName: String = s"$src-$tgt"

  /** Sequence of files to download as part of this dataset. */
  override def filesToDownload: Seq[String] = Seq(s"${IWSLT16DatasetLoader.url}/$src/$tgt/$directoryName.tgz")

  /** Returns all the corpora (tuples containing name, source file, target file, and a file processor to use)
    * of this dataset type. */
  override def corpora(datasetType: DatasetType): Seq[(String, File, File, FileProcessor)] = datasetType match {
    case Train => Seq(("train",
        File(downloadsDir) / directoryName / directoryName / s"train.tags.$directoryName.$src",
        File(downloadsDir) / directoryName / directoryName / s"train.tags.$directoryName.$tgt",
        TEDConverter))
    case Dev => Seq(("dev2010",
        File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.dev2010.$directoryName.$src.xml",
        File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.dev2010.$directoryName.$tgt.xml",
        SGMConverter))
    case Test => Seq(
      ("tst2010",
          File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2010.$directoryName.$src.xml",
          File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2010.$directoryName.$tgt.xml",
          SGMConverter),
      ("tst2011",
          File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2011.$directoryName.$src.xml",
          File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2011.$directoryName.$tgt.xml",
          SGMConverter),
      ("tst2012",
          File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2012.$directoryName.$src.xml",
          File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2012.$directoryName.$tgt.xml",
          SGMConverter),
      ("tst2013",
          File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2013.$directoryName.$src.xml",
          File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2013.$directoryName.$tgt.xml",
          SGMConverter),
      ("tst2014",
          File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2014.$directoryName.$src.xml",
          File(downloadsDir) / directoryName / directoryName / s"IWSLT16.TED.tst2014.$directoryName.$tgt.xml",
          SGMConverter))
  }
}

object IWSLT16DatasetLoader {
  val url: String = "https://wit3.fbk.eu/archive/2016-01/texts"

  val supportedLanguagePairs: Set[(Language, Language)] = Set(
    (english, arabic), (english, czech), (english, german), (english, french))

  def isLanguagePairSupported(srcLanguage: Language, tgtLanguage: Language): Boolean = {
    supportedLanguagePairs.contains((srcLanguage, tgtLanguage)) ||
        supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  def apply(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataConfig: DataConfig
  ): IWSLT16DatasetLoader = {
    new IWSLT16DatasetLoader(srcLanguage, tgtLanguage, dataConfig)
  }
}
