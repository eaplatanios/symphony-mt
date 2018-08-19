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
class WMT16Loader(
    override val srcLanguage: Language,
    override val tgtLanguage: Language,
    val config: DataConfig
) extends ParallelDatasetLoader(srcLanguage = srcLanguage, tgtLanguage = tgtLanguage) {
  require(
    WMT16Loader.isLanguagePairSupported(srcLanguage, tgtLanguage),
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
    WMT16Loader.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  protected def commonCrawlDataset: Option[CommonCrawlLoader] = {
    if (CommonCrawlLoader.isLanguagePairSupported(srcLanguage, tgtLanguage))
      Some(CommonCrawlLoader(srcLanguage, tgtLanguage, dataConfig.copy(workingDir = config.workingDir)))
    else
      None
  }

  protected def europarlV7Dataset: Option[EuroparlV7Loader] = {
    if (EuroparlV7Loader.isLanguagePairSupported(srcLanguage, tgtLanguage))
      Some(EuroparlV7Loader(srcLanguage, tgtLanguage, dataConfig.copy(workingDir = config.workingDir)))
    else
      None
  }

  protected def europarlV8Dataset: Option[EuroparlV8Loader] = {
    if (EuroparlV8Loader.isLanguagePairSupported(srcLanguage, tgtLanguage))
      Some(EuroparlV8Loader(srcLanguage, tgtLanguage, dataConfig.copy(workingDir = config.workingDir)))
    else
      None
  }

  protected def newsCommentaryV11Dataset: Option[NewsCommentaryV11Loader] = {
    if (NewsCommentaryV11Loader.isLanguagePairSupported(srcLanguage, tgtLanguage))
      Some(NewsCommentaryV11Loader(srcLanguage, tgtLanguage, dataConfig.copy(workingDir = config.workingDir)))
    else
      None
  }

  // TODO: Add support for the "CzEng 1.6pre" dataset.
  // TODO: Add support for the "Yandex Corpus" dataset.
  // TODO: Add support for the "Wiki Headlines" dataset.
  // TODO: Add support for the "SETIMES2" dataset.

  /** Sequence of files to download as part of this dataset. */
  override def filesToDownload: Seq[String] = {
    WMT16Loader.devArchives.map(archive => s"${WMT16Loader.newsCommentaryUrl}/$archive.tgz") ++
        WMT16Loader.testArchives.map(archive => s"${WMT16Loader.newsCommentaryUrl}/$archive.tgz")
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
        WMT16Loader.devArchives.foreach(archive => {
          (File(downloadsDir) / archive)
              .copyTo(File(downloadsDir) / "dev", overwrite = true)
        })
        var corpora = Seq.empty[(ParallelDataset.Tag, File, File, FileProcessor)]
        val supported2008Languages = Set[Language](Czech, English, French, German, Hungarian, Spanish)
        if (supported2008Languages.contains(srcLanguage) && supported2008Languages.contains(tgtLanguage))
          corpora ++= Seq((WMT16Loader.NewsTest2008,
              File(downloadsDir) / "dev" / "dev" / s"news-test2008-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"news-test2008-ref.$tgt.sgm",
              SGMConverter >> Normalizer))
        val supported2009Languages = Set[Language](Czech, English, French, German, Hungarian, Italian, Spanish)
        if (supported2009Languages.contains(srcLanguage) && supported2009Languages.contains(tgtLanguage))
          corpora ++= Seq((WMT16Loader.NewsTest2009,
              File(downloadsDir) / "dev" / "dev" / s"newstest2009-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newstest2009-ref.$tgt.sgm",
              SGMConverter >> Normalizer))
        val supported2009SysCombLanguages = Set[Language](Czech, English, French, German, Hungarian, Italian, Spanish)
        if (supported2009SysCombLanguages.contains(srcLanguage) && supported2009SysCombLanguages.contains(tgtLanguage))
          corpora ++= Seq((WMT16Loader.NewsSysComb2009,
              File(downloadsDir) / "dev" / "dev" / s"newssyscomb2009-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newssyscomb2009-ref.$tgt.sgm",
              SGMConverter >> Normalizer))
        val supported2010Languages = Set[Language](Czech, English, French, German, Spanish)
        if (supported2010Languages.contains(srcLanguage) && supported2010Languages.contains(tgtLanguage))
          corpora ++= Seq((WMT16Loader.NewsTest2010,
              File(downloadsDir) / "dev" / "dev" / s"newstest2010-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newstest2010-ref.$tgt.sgm",
              SGMConverter >> Normalizer))
        val supported2011Languages = Set[Language](Czech, English, French, German, Spanish)
        if (supported2011Languages.contains(srcLanguage) && supported2011Languages.contains(tgtLanguage))
          corpora ++= Seq((WMT16Loader.NewsTest2011,
              File(downloadsDir) / "dev" / "dev" / s"newstest2011-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newstest2011-ref.$tgt.sgm",
              SGMConverter >> Normalizer))
        val supported2012Languages = Set[Language](Czech, English, French, German, Russian, Spanish)
        if (supported2012Languages.contains(srcLanguage) && supported2012Languages.contains(tgtLanguage))
          corpora ++= Seq((WMT16Loader.NewsTest2012,
              File(downloadsDir) / "dev" / "dev" / s"newstest2012-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newstest2012-ref.$tgt.sgm",
              SGMConverter >> Normalizer))
        val supported2013Languages = Set[Language](Czech, English, French, German, Russian, Spanish)
        if (supported2013Languages.contains(srcLanguage) && supported2013Languages.contains(tgtLanguage))
          corpora ++= Seq((WMT16Loader.NewsTest2013,
              File(downloadsDir) / "dev" / "dev" / s"newstest2013-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newstest2013-ref.$tgt.sgm",
              SGMConverter >> Normalizer))
        val pair = if (reversed) s"$tgt$src" else s"$src$tgt"
        val supported2014Languages = Set[Language](Czech, English, French, German, Hindi, Russian)
        if (supported2014Languages.contains(srcLanguage) && supported2014Languages.contains(tgtLanguage))
          corpora ++= Seq((WMT16Loader.NewsTest2014,
              File(downloadsDir) / "dev" / "dev" / s"newstest2014-$pair-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newstest2014-$pair-ref.$tgt.sgm",
              SGMConverter >> Normalizer))
        val supported2014DevLanguages = Set[Language](English, Hindi)
        if (supported2014DevLanguages.contains(srcLanguage) && supported2014DevLanguages.contains(tgtLanguage))
          corpora ++= Seq((WMT16Loader.NewsDev2014,
              File(downloadsDir) / "dev" / "dev" / s"newsdev2014-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newsdev2014-ref.$tgt.sgm",
              SGMConverter >> Normalizer))
        val supported2015Languages = Set[Language](Czech, English, Finnish, German, Russian)
        if (supported2015Languages.contains(srcLanguage) && supported2015Languages.contains(tgtLanguage))
          corpora ++= Seq((WMT16Loader.NewsTest2015,
              File(downloadsDir) / "dev" / "dev" / s"newstest2015-$src$tgt-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newstest2015-$src$tgt-ref.$tgt.sgm",
              SGMConverter >> Normalizer))
        val supported2015DevLanguages = Set[Language](English, Finnish)
        if (supported2015DevLanguages.contains(srcLanguage) && supported2015DevLanguages.contains(tgtLanguage))
          corpora ++= Seq((WMT16Loader.NewsDev2015,
              File(downloadsDir) / "dev" / "dev" / s"newsdev2015-$src$tgt-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newsdev2015-$src$tgt-ref.$tgt.sgm",
              SGMConverter >> Normalizer))
        val supported2015DiscussDevLanguages = Set[Language](English, French)
        if (supported2015DiscussDevLanguages.contains(srcLanguage) &&
            supported2015DiscussDevLanguages.contains(tgtLanguage))
          corpora ++= Seq((WMT16Loader.NewsDiscussDev2015,
              File(downloadsDir) / "dev" / "dev" / s"newsdiscussdev2015-$src$tgt-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newsdiscussdev2015-$src$tgt-ref.$tgt.sgm",
              SGMConverter >> Normalizer))
        val supported2015DiscussTestLanguages = Set[Language](English, French)
        if (supported2015DiscussTestLanguages.contains(srcLanguage) &&
            supported2015DiscussTestLanguages.contains(tgtLanguage))
          corpora ++= Seq((WMT16Loader.NewsDiscussTest2015,
              File(downloadsDir) / "dev" / "dev" / s"newsdiscusstest2015-$src$tgt-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newsdiscusstest2015-$src$tgt-ref.$tgt.sgm",
              SGMConverter >> Normalizer))
        val supported2016DevLanguages = Set[Language](English, Romanian, Turkish)
        if (supported2016DevLanguages.contains(srcLanguage) && supported2016DevLanguages.contains(tgtLanguage))
          corpora ++= Seq((WMT16Loader.NewsDev2016,
              File(downloadsDir) / "dev" / "dev" / s"newsdev2016-$src$tgt-src.$src.sgm",
              File(downloadsDir) / "dev" / "dev" / s"newsdev2016-$src$tgt-ref.$tgt.sgm",
              SGMConverter >> Normalizer))
        corpora
      case Test =>
        WMT16Loader.testArchives.foreach(archive => {
          (File(downloadsDir) / archive)
              .copyTo(File(downloadsDir) / "test", overwrite = true)
        })
        var corpora = Seq.empty[(ParallelDataset.Tag, File, File, FileProcessor)]
        val supported2016Languages = Set[Language](Czech, English, Finnish, German, Romanian, Russian, Turkish)
        if (supported2016Languages.contains(srcLanguage) && supported2016Languages.contains(tgtLanguage))
          corpora ++= Seq((WMT16Loader.NewsTest2016,
              File(downloadsDir) / "test" / "test" / s"newstest2016-$src$tgt-src.$src.sgm",
              File(downloadsDir) / "test" / "test" / s"newstest2016-$src$tgt-ref.$tgt.sgm",
              SGMConverter >> Normalizer))
        corpora
    }
  }
}

object WMT16Loader {
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
  ): WMT16Loader = {
    new WMT16Loader(srcLanguage, tgtLanguage, dataConfig)
  }

  trait Tag extends ParallelDataset.Tag

  object Tag {
    @throws[IllegalArgumentException]
    def fromName(name: String): Tag = name match {
      case "newstest2008" => NewsTest2008
      case "newstest2009" => NewsTest2009
      case "newssyscomb2009" => NewsSysComb2009
      case "newstest2010" => NewsTest2010
      case "newstest2011" => NewsTest2011
      case "newstest2012" => NewsTest2012
      case "newstest2013" => NewsTest2013
      case "newstest2014" => NewsTest2014
      case "newsdev2014" => NewsDev2014
      case "newstest2015" => NewsTest2015
      case "newsdev2015" => NewsDev2015
      case "newsdiscussdev2015" => NewsDiscussDev2015
      case "newsdiscusstest2015" => NewsDiscussTest2015
      case "newsdev2016" => NewsDev2016
      case "newstest2016" => NewsTest2016
      case _ => throw new IllegalArgumentException(s"'$name' is not a valid WMT-16 tag.")
    }
  }

  case object NewsTest2008 extends Tag {
    override val value: String = "newstest2008"
  }

  case object NewsTest2009 extends Tag {
    override val value: String = "newstest2009"
  }

  case object NewsSysComb2009 extends Tag {
    override val value: String = "newssyscomb2009"
  }

  case object NewsTest2010 extends Tag {
    override val value: String = "newstest2010"
  }

  case object NewsTest2011 extends Tag {
    override val value: String = "newstest2011"
  }

  case object NewsTest2012 extends Tag {
    override val value: String = "newstest2012"
  }

  case object NewsTest2013 extends Tag {
    override val value: String = "newstest2013"
  }

  case object NewsTest2014 extends Tag {
    override val value: String = "newstest2014"
  }

  case object NewsDev2014 extends Tag {
    override val value: String = "newsdev2014"
  }

  case object NewsTest2015 extends Tag {
    override val value: String = "newstest2015"
  }

  case object NewsDev2015 extends Tag {
    override val value: String = "newsdev2015"
  }

  case object NewsDiscussDev2015 extends Tag {
    override val value: String = "newsdiscussdev2015"
  }

  case object NewsDiscussTest2015 extends Tag {
    override val value: String = "newsdiscusstest2015"
  }

  case object NewsDev2016 extends Tag {
    override val value: String = "newsdev2016"
  }

  case object NewsTest2016 extends Tag {
    override val value: String = "newstest2016"
  }
}
