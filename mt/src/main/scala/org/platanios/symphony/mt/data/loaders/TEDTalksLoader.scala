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
import org.platanios.symphony.mt.data.processors.{FileProcessor, TSVConverter}

import better.files._

import java.nio.file.Path

/**
  * @author Emmanouil Antonios Platanios
  */
class TEDTalksLoader(
    override val srcLanguage: Language,
    override val tgtLanguage: Language,
    val config: DataConfig
) extends ParallelDatasetLoader(srcLanguage, tgtLanguage) {
  require(
    TEDTalksLoader.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the TED-Talks dataset.")

  override def name: String = "TED-Talks"

  override def dataConfig: DataConfig = {
    config.copy(workingDir =
        config.workingDir
            .resolve("ted-talks")
            .resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"))
  }

  override def downloadsDir: Path = config.workingDir.resolve("ted-talks").resolve("downloads")

  /** Sequence of files to download as part of this dataset. */
  override def filesToDownload: Seq[String] = Seq(s"${TEDTalksLoader.url}/${TEDTalksLoader.filename}.tar.gz")

  /** Returns all the corpora (tuples containing tag, source file, target file, and a file processor to use)
    * of this dataset type. */
  override def corpora(datasetType: DatasetType): Seq[(ParallelDataset.Tag, File, File, FileProcessor)] = {
    datasetType match {
      case Train => Seq((TEDTalksLoader.Train,
          File(downloadsDir) / TEDTalksLoader.filename / TEDTalksLoader.filename / "all_talks_train.tsv",
          File(downloadsDir) / TEDTalksLoader.filename / TEDTalksLoader.filename / "all_talks_train.tsv",
          TSVConverter))
      case Dev => Seq((TEDTalksLoader.Dev,
          File(downloadsDir) / TEDTalksLoader.filename / TEDTalksLoader.filename / "all_talks_dev.tsv",
          File(downloadsDir) / TEDTalksLoader.filename / TEDTalksLoader.filename / "all_talks_dev.tsv",
          TSVConverter))
      case Test => Seq((TEDTalksLoader.Test,
          File(downloadsDir) / TEDTalksLoader.filename / TEDTalksLoader.filename / "all_talks_test.tsv",
          File(downloadsDir) / TEDTalksLoader.filename / TEDTalksLoader.filename / "all_talks_test.tsv",
          TSVConverter))
    }
  }
}

object TEDTalksLoader {
  val url: String = "http://phontron.com/data"
  val filename: String = "ted_talks"

  val supportedLanguagePairs: Set[(Language, Language)] = Set(
    Albanian, Arabic, Armenian, Azerbaijani, Basque, Belarusian, Bengali, Bosnian, Bulgarian, Burmese, Chinese,
    ChineseMainland, ChineseTaiwan, Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Finnish, French,
    FrenchCanada, Galician, Georgian, German, Greek, Hebrew, Hindi, Hungarian, Indonesian, Italian, Japanese, Kazakh,
    Korean, Kurdish, Lithuanian, Macedonian, Malay, Marathi, Mongolian, Norwegian, Persian, Polish, Portuguese,
    PortugueseBrazil, Romanian, Russian, Tamil, Thai, Serbian, Slovak, Slovenian, Swedish, Spanish, Turkish, Ukranian,
    Urdu, Vietnamese).toSeq.combinations(2).map(p => (p(0), p(1))).toSet

  def isLanguagePairSupported(srcLanguage: Language, tgtLanguage: Language): Boolean = {
    supportedLanguagePairs.contains((srcLanguage, tgtLanguage)) ||
        supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  def apply(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataConfig: DataConfig
  ): TEDTalksLoader = {
    new TEDTalksLoader(srcLanguage, tgtLanguage, dataConfig)
  }

  trait Tag extends ParallelDataset.Tag

  object Tag {
    @throws[IllegalArgumentException]
    def fromName(name: String): Tag = name match {
      case "train" => Train
      case "dev" => Dev
      case "test" => Test
      case _ => throw new IllegalArgumentException(s"'$name' is not a valid TED-Talks tag.")
    }
  }

  case object Train extends Tag {
    override val value: String = "train"
  }

  case object Dev extends Tag {
    override val value: String = "dev"
  }

  case object Test extends Tag {
    override val value: String = "test"
  }
}
