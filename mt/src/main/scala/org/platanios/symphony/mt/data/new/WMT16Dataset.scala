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
import org.platanios.symphony.mt.Language._

import better.files._
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.Path

/**
  * @author Emmanouil Antonios Platanios
  */
class WMT16Dataset(
    val srcLanguage: Language,
    val tgtLanguage: Language,
    override protected val workingDir: Path,
    override val bufferSize: Int = 8192,
    override val tokenize: Boolean = true
) extends Dataset(
  workingDir = workingDir.resolve("wmt-16").resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"),
  bufferSize = bufferSize,
  tokenize = tokenize
)(
  downloadsDir = workingDir.resolve("wmt-16").resolve("downloads")
) {
  require(
    WMT16Dataset.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the WMT16 dataset.")

  override def dataDir: Path = {
    workingDir.resolve("wmt-16").resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}")
  }

  private[this] val src: String = srcLanguage.abbreviation
  private[this] val tgt: String = tgtLanguage.abbreviation

  private[this] val reversed: Boolean = {
    WMT16Dataset.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  protected def commonCrawlDataset: Option[CommonCrawlDataset] = {
    if (CommonCrawlDataset.isLanguagePairSupported(srcLanguage, tgtLanguage))
      Some(CommonCrawlDataset(srcLanguage, tgtLanguage, workingDir, bufferSize))
    else
      None
  }

  protected def europarlV7Dataset: Option[EuroparlV7Dataset] = {
    if (EuroparlV7Dataset.isLanguagePairSupported(srcLanguage, tgtLanguage))
      Some(EuroparlV7Dataset(srcLanguage, tgtLanguage, workingDir, bufferSize))
    else
      None
  }

  protected def europarlV8Dataset: Option[EuroparlV8Dataset] = {
    if (EuroparlV8Dataset.isLanguagePairSupported(srcLanguage, tgtLanguage))
      Some(EuroparlV8Dataset(srcLanguage, tgtLanguage, workingDir, bufferSize))
    else
      None
  }

  protected def newsCommentaryV11Dataset: Option[NewsCommentaryV11Dataset] = {
    if (NewsCommentaryV11Dataset.isLanguagePairSupported(srcLanguage, tgtLanguage))
      Some(NewsCommentaryV11Dataset(srcLanguage, tgtLanguage, workingDir, bufferSize))
    else
      None
  }

  // TODO: Add support for the "CzEng 1.6pre" dataset.
  // TODO: Add support for the "Yandex Corpus" dataset.
  // TODO: Add support for the "Wiki Headlines" dataset.
  // TODO: Add support for the "SETIMES2" dataset.

  /** Sequence of files to download as part of this dataset. */
  override def filesToDownload: Seq[String] = {
    commonCrawlDataset.map(_.filesToDownload).getOrElse(Seq.empty) ++
        europarlV7Dataset.map(_.filesToDownload).getOrElse(Seq.empty) ++
        europarlV8Dataset.map(_.filesToDownload).getOrElse(Seq.empty) ++
        newsCommentaryV11Dataset.map(_.filesToDownload).getOrElse(Seq.empty) ++
        WMT16Dataset.devArchives.map(archive => s"${WMT16Dataset.newsCommentaryUrl}/$archive.tgz") ++
        WMT16Dataset.testArchives.map(archive => s"${WMT16Dataset.newsCommentaryUrl}/$archive.tgz")
  }

  /** Grouped files included in this dataset. */
  override def groupedFiles: Dataset.GroupedFiles = {
    WMT16Dataset.devArchives.foreach(archive => {
      (File(dataDir) / archive)
          .copyTo(File(dataDir) / "dev", overwrite = true)
    })
    WMT16Dataset.testArchives.foreach(archive => {
      (File(dataDir) / archive)
          .copyTo(File(dataDir) / "test", overwrite = true)
    })
    Dataset.GroupedFiles(
      trainCorpora = {
        commonCrawlDataset.map(_.groupedFiles.trainCorpora).getOrElse(Seq.empty) ++
            europarlV7Dataset.map(_.groupedFiles.trainCorpora).getOrElse(Seq.empty) ++
            europarlV8Dataset.map(_.groupedFiles.trainCorpora).getOrElse(Seq.empty) ++
            newsCommentaryV11Dataset.map(_.groupedFiles.trainCorpora).getOrElse(Seq.empty)
      },
      devCorpora = {
        val supported2008Languages = Set[Language](Czech, English, French, German, Hungarian, Spanish)
        if (supported2008Languages.contains(srcLanguage) && supported2008Languages.contains(tgtLanguage))
          Seq(("WMT16/newstest2008",
              File(dataDir) / s"news-test2008-src.$src",
              File(dataDir) / s"news-test2008-ref.$tgt"))
        else
          Seq.empty
      } ++ {
        val supported2009Languages = Set[Language](Czech, English, French, German, Hungarian, Italian, Spanish)
        if (supported2009Languages.contains(srcLanguage) && supported2009Languages.contains(tgtLanguage))
          Seq(("WMT16/newstest2009",
              File(dataDir) / s"newstest2009-src.$src",
              File(dataDir) / s"newstest2009-ref.$tgt"))
        else
          Seq.empty
      } ++ {
        val supported2009SysCombLanguages = Set[Language](Czech, English, French, German, Hungarian, Italian, Spanish)
        if (supported2009SysCombLanguages.contains(srcLanguage) && supported2009SysCombLanguages.contains(tgtLanguage))
          Seq(("WMT16/newssyscomb2009",
              File(dataDir) / s"newssyscomb2009-src.$src",
              File(dataDir) / s"newssyscomb2009-ref.$tgt"))
        else
          Seq.empty
      } ++ {
        val supported2010Languages = Set[Language](Czech, English, French, German, Spanish)
        if (supported2010Languages.contains(srcLanguage) && supported2010Languages.contains(tgtLanguage))
          Seq(("WMT16/newstest2010",
              File(dataDir) / s"newstest2010-src.$src",
              File(dataDir) / s"newstest2010-ref.$tgt"))
        else
          Seq.empty
      } ++ {
        val supported2011Languages = Set[Language](Czech, English, French, German, Spanish)
        if (supported2011Languages.contains(srcLanguage) && supported2011Languages.contains(tgtLanguage))
          Seq(("WMT16/newstest2011",
              File(dataDir) / s"newstest2011-src.$src",
              File(dataDir) / s"newstest2011-ref.$tgt"))
        else
          Seq.empty
      } ++ {
        val supported2012Languages = Set[Language](Czech, English, French, German, Russian, Spanish)
        if (supported2012Languages.contains(srcLanguage) && supported2012Languages.contains(tgtLanguage))
          Seq(("WMT16/newstest2012",
              File(dataDir) / s"newstest2012-src.$src",
              File(dataDir) / s"newstest2012-ref.$tgt"))
        else
          Seq.empty
      } ++ {
        val supported2013Languages = Set[Language](Czech, English, French, German, Russian, Spanish)
        if (supported2013Languages.contains(srcLanguage) && supported2013Languages.contains(tgtLanguage))
          Seq(("WMT16/newstest2013",
              File(dataDir) / s"newstest2013-src.$src",
              File(dataDir) / s"newstest2013-ref.$tgt"))
        else
          Seq.empty
      } ++ {
        val pair = if (reversed) s"$tgt$src" else s"$src$tgt"
        val supported2014Languages = Set[Language](Czech, English, French, German, Hindi, Russian)
        if (supported2014Languages.contains(srcLanguage) && supported2014Languages.contains(tgtLanguage))
          Seq(("WMT16/newstest2014",
              File(dataDir) / s"newstest2014-$pair-src.$src",
              File(dataDir) / s"newstest2014-$pair-ref.$tgt"))
        else
          Seq.empty
      } ++ {
        val supported2014DevLanguages = Set[Language](English, Hindi)
        if (supported2014DevLanguages.contains(srcLanguage) && supported2014DevLanguages.contains(tgtLanguage))
          Seq(("WMT16/newsdev2014",
              File(dataDir) / s"newsdev2014-src.$src",
              File(dataDir) / s"newsdev2014-ref.$tgt"))
        else
          Seq.empty
      } ++ {
        val supported2015Languages = Set[Language](Czech, English, Finnish, German, Russian)
        if (supported2015Languages.contains(srcLanguage) && supported2015Languages.contains(tgtLanguage))
          Seq(("WMT16/newstest2015",
              File(dataDir) / s"newstest2015-$src$tgt-src.$src",
              File(dataDir) / s"newstest2015-$src$tgt-ref.$tgt"))
        else
          Seq.empty
      } ++ {
        val supported2015DevLanguages = Set[Language](English, Finnish)
        if (supported2015DevLanguages.contains(srcLanguage) && supported2015DevLanguages.contains(tgtLanguage))
          Seq(("WMT16/newsdev2015",
              File(dataDir) / s"newsdev2015-$src$tgt-src.$src",
              File(dataDir) / s"newsdev2015-$src$tgt-ref.$tgt"))
        else
          Seq.empty
      } ++ {
        val supported2015DiscussDevLanguages = Set[Language](English, French)
        if (supported2015DiscussDevLanguages.contains(srcLanguage) &&
            supported2015DiscussDevLanguages.contains(tgtLanguage))
          Seq(("WMT16/newsdiscussdev2015",
              File(dataDir) / s"newsdiscussdev2015-$src$tgt-src.$src",
              File(dataDir) / s"newsdiscussdev2015-$src$tgt-ref.$tgt"))
        else
          Seq.empty
      } ++ {
        val supported2015DiscussTestLanguages = Set[Language](English, French)
        if (supported2015DiscussTestLanguages.contains(srcLanguage) &&
            supported2015DiscussTestLanguages.contains(tgtLanguage))
          Seq(("WMT16/newsdiscusstest2015",
              File(dataDir) / s"newsdiscusstest2015-$src$tgt-src.$src",
              File(dataDir) / s"newsdiscusstest2015-$src$tgt-ref.$tgt"))
        else
          Seq.empty
      } ++ {
        val supported2016DevLanguages = Set[Language](English, Romanian, Turkish)
        if (supported2016DevLanguages.contains(srcLanguage) && supported2016DevLanguages.contains(tgtLanguage))
          Seq(("WMT16/newsdev2016",
              File(dataDir) / s"newsdev2016-$src$tgt-src.$src",
              File(dataDir) / s"newsdev2016-$src$tgt-ref.$tgt"))
        else
          Seq.empty
      },
      testCorpora = {
        // 2016 Data
        val supported2016Languages = Set[Language](Czech, English, Finnish, German, Romanian, Russian, Turkish)
        if (supported2016Languages.contains(srcLanguage) && supported2016Languages.contains(tgtLanguage))
          Seq(("WMT16/newstest2016",
              File(dataDir) / s"newstest2016-$src$tgt-src.$src",
              File(dataDir) / s"newstest2016-$src$tgt-ref.$tgt"))
        else
          Seq.empty
      })
  }
}

object WMT16Dataset {
  private[WMT16Dataset] val logger = Logger(LoggerFactory.getLogger("WMT16 Dataset"))

  val newsCommentaryUrl            : String      = "http://data.statmt.org/wmt16/translation-task"
  val newsCommentaryParallelArchive: String      = "training-parallel-nc-v11"
  val devArchives                  : Seq[String] = Seq("dev", "dev-romanian-updated")
  val testArchives                 : Seq[String] = Seq("test")

  val supportedLanguagePairs: Set[(Language, Language)] = Set(
    (Czech, English), (Finnish, English), (German, English),
    (Romanian, English), (Russian, English), (Turkish, English))

  def isLanguagePairSupported(srcLanguage: Language, tgtLanguage: Language): Boolean = {
    supportedLanguagePairs.contains((srcLanguage, tgtLanguage)) ||
        supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  def apply(
      srcLanguage: Language,
      tgtLanguage: Language,
      workingDir: Path,
      bufferSize: Int = 8192,
      tokenize: Boolean = true
  ): WMT16Dataset = {
    new WMT16Dataset(srcLanguage, tgtLanguage, workingDir, bufferSize, tokenize)
  }
}
