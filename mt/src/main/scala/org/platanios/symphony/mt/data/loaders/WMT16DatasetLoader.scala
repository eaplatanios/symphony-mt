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
import org.platanios.symphony.mt.data.processors.{FileProcessor, SGMConverter}

import better.files._

import java.nio.file.Path

/**
  * @author Emmanouil Antonios Platanios
  */
class WMT16DatasetLoader(
    override val srcLanguage: Language,
    override val tgtLanguage: Language,
    val config: DataConfig
) extends ParallelDatasetLoader(srcLanguage = srcLanguage, tgtLanguage = tgtLanguage) {
  require(
    WMT16DatasetLoader.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the WMT16 dataset.")

  override def name: String = "WMT-16"

  override def dataConfig: DataConfig = {
    config.copy(workingDir =
        config.workingDir
            .resolve("wmt-16")
            .resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"))
  }

  override def downloadsDir: Path = config.workingDir.resolve("wmt-16").resolve("downloads")

  private[this] def reversed: Boolean = {
    WMT16DatasetLoader.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  protected def commonCrawlDataset: Option[CommonCrawlDatasetLoader] = {
    if (CommonCrawlDatasetLoader.isLanguagePairSupported(srcLanguage, tgtLanguage))
      Some(CommonCrawlDatasetLoader(srcLanguage, tgtLanguage, dataConfig.copy(workingDir = config.workingDir)))
    else
      None
  }

  protected def europarlV7Dataset: Option[EuroparlV7DatasetLoader] = {
    if (EuroparlV7DatasetLoader.isLanguagePairSupported(srcLanguage, tgtLanguage))
      Some(EuroparlV7DatasetLoader(srcLanguage, tgtLanguage, dataConfig.copy(workingDir = config.workingDir)))
    else
      None
  }

  protected def europarlV8Dataset: Option[EuroparlV8DatasetLoader] = {
    if (EuroparlV8DatasetLoader.isLanguagePairSupported(srcLanguage, tgtLanguage))
      Some(EuroparlV8DatasetLoader(srcLanguage, tgtLanguage, dataConfig.copy(workingDir = config.workingDir)))
    else
      None
  }

  protected def newsCommentaryV11Dataset: Option[NewsCommentaryV11DatasetLoader] = {
    if (NewsCommentaryV11DatasetLoader.isLanguagePairSupported(srcLanguage, tgtLanguage))
      Some(NewsCommentaryV11DatasetLoader(srcLanguage, tgtLanguage, dataConfig.copy(workingDir = config.workingDir)))
    else
      None
  }

  // TODO: Add support for the "CzEng 1.6pre" dataset.
  // TODO: Add support for the "Yandex Corpus" dataset.
  // TODO: Add support for the "Wiki Headlines" dataset.
  // TODO: Add support for the "SETIMES2" dataset.

  /** Sequence of files to download as part of this dataset. */
  override def filesToDownload: Seq[String] = {
    WMT16DatasetLoader.devArchives.map(archive => s"${WMT16DatasetLoader.newsCommentaryUrl}/$archive.tgz") ++
        WMT16DatasetLoader.testArchives.map(archive => s"${WMT16DatasetLoader.newsCommentaryUrl}/$archive.tgz")
  }

  /** Returns all the corpora (tuples containing tag, source file, target file, and a file processor to use)
    * of this dataset type. */
  override def corpora(datasetType: DatasetType): Seq[(ParallelDataset.Tag, File, File, FileProcessor)] = {
    datasetType match {
      case Train =>
        commonCrawlDataset.map(_.corpora(Train)).getOrElse(Seq.empty) ++
            europarlV7Dataset.map(_.corpora(Train)).getOrElse(Seq.empty) ++
            europarlV8Dataset.map(_.corpora(Train)).getOrElse(Seq.empty) ++
            newsCommentaryV11Dataset.map(_.corpora(Train)).getOrElse(Seq.empty)
      case Dev =>
        WMT16DatasetLoader.devArchives.foreach(archive => {
          (File(downloadsDir) / archive)
              .copyTo(File(downloadsDir) / "dev", overwrite = true)
        })
        var corpora = Seq.empty[(ParallelDataset.Tag, File, File, FileProcessor)]
        val supported2008Languages = Set[Language](Czech, English, French, German, Hungarian, Spanish)
        if (supported2008Languages.contains(srcLanguage) && supported2008Languages.contains(tgtLanguage))
          corpora ++= Seq((WMT16DatasetLoader.NewsTest2008,
              File(downloadsDir) / "dev" / "dev" / s"news-test2008-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"news-test2008-ref.$tgt.sgm", SGMConverter))
        val supported2009Languages = Set[Language](Czech, English, French, German, Hungarian, Italian, Spanish)
        if (supported2009Languages.contains(srcLanguage) && supported2009Languages.contains(tgtLanguage))
          corpora ++= Seq((WMT16DatasetLoader.NewsTest2009,
              File(downloadsDir) / "dev" / "dev" / s"newstest2009-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newstest2009-ref.$tgt.sgm", SGMConverter))
        val supported2009SysCombLanguages = Set[Language](Czech, English, French, German, Hungarian, Italian, Spanish)
        if (supported2009SysCombLanguages.contains(srcLanguage) && supported2009SysCombLanguages.contains(tgtLanguage))
          corpora ++= Seq((WMT16DatasetLoader.NewsSysComb2009,
              File(downloadsDir) / "dev" / "dev" / s"newssyscomb2009-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newssyscomb2009-ref.$tgt.sgm", SGMConverter))
        val supported2010Languages = Set[Language](Czech, English, French, German, Spanish)
        if (supported2010Languages.contains(srcLanguage) && supported2010Languages.contains(tgtLanguage))
          corpora ++= Seq((WMT16DatasetLoader.NewsTest2010,
              File(downloadsDir) / "dev" / "dev" / s"newstest2010-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newstest2010-ref.$tgt.sgm", SGMConverter))
        val supported2011Languages = Set[Language](Czech, English, French, German, Spanish)
        if (supported2011Languages.contains(srcLanguage) && supported2011Languages.contains(tgtLanguage))
          corpora ++= Seq((WMT16DatasetLoader.NewsTest2011,
              File(downloadsDir) / "dev" / "dev" / s"newstest2011-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newstest2011-ref.$tgt.sgm", SGMConverter))
        val supported2012Languages = Set[Language](Czech, English, French, German, Russian, Spanish)
        if (supported2012Languages.contains(srcLanguage) && supported2012Languages.contains(tgtLanguage))
          corpora ++= Seq((WMT16DatasetLoader.NewsTest2012,
              File(downloadsDir) / "dev" / "dev" / s"newstest2012-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newstest2012-ref.$tgt.sgm", SGMConverter))
        val supported2013Languages = Set[Language](Czech, English, French, German, Russian, Spanish)
        if (supported2013Languages.contains(srcLanguage) && supported2013Languages.contains(tgtLanguage))
          corpora ++= Seq((WMT16DatasetLoader.NewsTest2013,
              File(downloadsDir) / "dev" / "dev" / s"newstest2013-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newstest2013-ref.$tgt.sgm", SGMConverter))
        val pair = if (reversed) s"$tgt$src" else s"$src$tgt"
        val supported2014Languages = Set[Language](Czech, English, French, German, Hindi, Russian)
        if (supported2014Languages.contains(srcLanguage) && supported2014Languages.contains(tgtLanguage))
          corpora ++= Seq((WMT16DatasetLoader.NewsTest2014,
              File(downloadsDir) / "dev" / "dev" / s"newstest2014-$pair-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newstest2014-$pair-ref.$tgt.sgm", SGMConverter))
        val supported2014DevLanguages = Set[Language](English, Hindi)
        if (supported2014DevLanguages.contains(srcLanguage) && supported2014DevLanguages.contains(tgtLanguage))
          corpora ++= Seq((WMT16DatasetLoader.NewsDev2014,
              File(downloadsDir) / "dev" / "dev" / s"newsdev2014-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newsdev2014-ref.$tgt.sgm", SGMConverter))
        val supported2015Languages = Set[Language](Czech, English, Finnish, German, Russian)
        if (supported2015Languages.contains(srcLanguage) && supported2015Languages.contains(tgtLanguage))
          corpora ++= Seq((WMT16DatasetLoader.NewsTest2015,
              File(downloadsDir) / "dev" / "dev" / s"newstest2015-$src$tgt-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newstest2015-$src$tgt-ref.$tgt.sgm", SGMConverter))
        val supported2015DevLanguages = Set[Language](English, Finnish)
        if (supported2015DevLanguages.contains(srcLanguage) && supported2015DevLanguages.contains(tgtLanguage))
          corpora ++= Seq((WMT16DatasetLoader.NewsDev2015,
              File(downloadsDir) / "dev" / "dev" / s"newsdev2015-$src$tgt-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newsdev2015-$src$tgt-ref.$tgt.sgm", SGMConverter))
        val supported2015DiscussDevLanguages = Set[Language](English, French)
        if (supported2015DiscussDevLanguages.contains(srcLanguage) &&
            supported2015DiscussDevLanguages.contains(tgtLanguage))
          corpora ++= Seq((WMT16DatasetLoader.NewsDiscussDev2015,
              File(downloadsDir) / "dev" / "dev" / s"newsdiscussdev2015-$src$tgt-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newsdiscussdev2015-$src$tgt-ref.$tgt.sgm", SGMConverter))
        val supported2015DiscussTestLanguages = Set[Language](English, French)
        if (supported2015DiscussTestLanguages.contains(srcLanguage) &&
            supported2015DiscussTestLanguages.contains(tgtLanguage))
          corpora ++= Seq((WMT16DatasetLoader.NewsDiscussTest2015,
              File(downloadsDir) / "dev" / "dev" / s"newsdiscusstest2015-$src$tgt-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newsdiscusstest2015-$src$tgt-ref.$tgt.sgm", SGMConverter))
        val supported2016DevLanguages = Set[Language](English, Romanian, Turkish)
        if (supported2016DevLanguages.contains(srcLanguage) && supported2016DevLanguages.contains(tgtLanguage))
          corpora ++= Seq((WMT16DatasetLoader.NewsDev2016,
              File(downloadsDir) / "dev" / "dev" / s"newsdev2016-$src$tgt-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newsdev2016-$src$tgt-ref.$tgt.sgm", SGMConverter))
        corpora
      case Test =>
        WMT16DatasetLoader.testArchives.foreach(archive => {
          (File(downloadsDir) / archive)
              .copyTo(File(downloadsDir) / "test", overwrite = true)
        })
        var corpora = Seq.empty[(ParallelDataset.Tag, File, File, FileProcessor)]
        val supported2016Languages = Set[Language](Czech, English, Finnish, German, Romanian, Russian, Turkish)
        if (supported2016Languages.contains(srcLanguage) && supported2016Languages.contains(tgtLanguage))
          corpora ++= Seq((WMT16DatasetLoader.NewsTest2016,
              File(downloadsDir) / "test" / "test" / s"newstest2016-$src$tgt-src.$src.sgm",
              File(downloadsDir) / "test" / "test" / s"newstest2016-$src$tgt-ref.$tgt.sgm", SGMConverter))
        corpora
    }
  }
}

object WMT16DatasetLoader {
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
      dataConfig: DataConfig
  ): WMT16DatasetLoader = {
    new WMT16DatasetLoader(srcLanguage, tgtLanguage, dataConfig)
  }

  case object NewsTest2008 extends ParallelDataset.Tag {
    override val value: String = "wmt-16/newstest2008"
  }

  case object NewsTest2009 extends ParallelDataset.Tag {
    override val value: String = "wmt-16/newstest2009"
  }

  case object NewsSysComb2009 extends ParallelDataset.Tag {
    override val value: String = "wmt-16/newssyscomb2009"
  }

  case object NewsTest2010 extends ParallelDataset.Tag {
    override val value: String = "wmt-16/newstest2010"
  }

  case object NewsTest2011 extends ParallelDataset.Tag {
    override val value: String = "wmt-16/newstest2011"
  }

  case object NewsTest2012 extends ParallelDataset.Tag {
    override val value: String = "wmt-16/newstest2012"
  }

  case object NewsTest2013 extends ParallelDataset.Tag {
    override val value: String = "wmt-16/newstest2013"
  }

  case object NewsTest2014 extends ParallelDataset.Tag {
    override val value: String = "wmt-16/newstest2014"
  }

  case object NewsDev2014 extends ParallelDataset.Tag {
    override val value: String = "wmt-16/newsdev2014"
  }

  case object NewsTest2015 extends ParallelDataset.Tag {
    override val value: String = "wmt-16/newstest2015"
  }

  case object NewsDev2015 extends ParallelDataset.Tag {
    override val value: String = "wmt-16/newsdev2015"
  }

  case object NewsDiscussDev2015 extends ParallelDataset.Tag {
    override val value: String = "wmt-16/newsdiscussdev2015"
  }

  case object NewsDiscussTest2015 extends ParallelDataset.Tag {
    override val value: String = "wmt-16/newsdiscusstest2015"
  }

  case object NewsDev2016 extends ParallelDataset.Tag {
    override val value: String = "wmt-16/newsdev2016"
  }

  case object NewsTest2016 extends ParallelDataset.Tag {
    override val value: String = "wmt-16/newstest2016"
  }
}
