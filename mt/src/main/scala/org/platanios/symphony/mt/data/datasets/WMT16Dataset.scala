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

package org.platanios.symphony.mt.data.datasets

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.Language._
import org.platanios.symphony.mt.data.Dataset

import better.files._

import java.nio.file.Path

/**
  * @author Emmanouil Antonios Platanios
  */
class WMT16Dataset(
    override protected val workingDir: Path,
    override val srcLanguage: Language,
    override val tgtLanguage: Language,
    override val bufferSize: Int = 8192,
    override val tokenize: Boolean = true,
    override val trainDataSentenceLengthBounds: (Int, Int) = (1, 80)
) extends Dataset(
  workingDir = workingDir.resolve("wmt-16").resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"),
  srcLanguage = srcLanguage,
  tgtLanguage = tgtLanguage,
  bufferSize = bufferSize,
  tokenize = tokenize,
  trainDataSentenceLengthBounds = trainDataSentenceLengthBounds
)(
  downloadsDir = workingDir.resolve("wmt-16")
) {
  require(
    WMT16Dataset.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the WMT16 dataset.")

  override def name: String = "WMT-16"

  private[this] def reversed: Boolean = {
    WMT16Dataset.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  protected def commonCrawlDataset: Option[CommonCrawlDataset] = {
    if (CommonCrawlDataset.isLanguagePairSupported(srcLanguage, tgtLanguage))
      Some(CommonCrawlDataset(workingDir, srcLanguage, tgtLanguage, bufferSize, tokenize = true))
    else
      None
  }

  protected def europarlV7Dataset: Option[EuroparlV7Dataset] = {
    if (EuroparlV7Dataset.isLanguagePairSupported(srcLanguage, tgtLanguage))
      Some(EuroparlV7Dataset(workingDir, srcLanguage, tgtLanguage, bufferSize, tokenize = true))
    else
      None
  }

  protected def europarlV8Dataset: Option[EuroparlV8Dataset] = {
    if (EuroparlV8Dataset.isLanguagePairSupported(srcLanguage, tgtLanguage))
      Some(EuroparlV8Dataset(workingDir, srcLanguage, tgtLanguage, bufferSize, tokenize = true))
    else
      None
  }

  protected def newsCommentaryV11Dataset: Option[NewsCommentaryV11Dataset] = {
    if (NewsCommentaryV11Dataset.isLanguagePairSupported(srcLanguage, tgtLanguage))
      Some(NewsCommentaryV11Dataset(workingDir, srcLanguage, tgtLanguage, bufferSize, tokenize = true))
    else
      None
  }

  // TODO: Add support for the "CzEng 1.6pre" dataset.
  // TODO: Add support for the "Yandex Corpus" dataset.
  // TODO: Add support for the "Wiki Headlines" dataset.
  // TODO: Add support for the "SETIMES2" dataset.

  /** Sequence of files to download as part of this dataset. */
  override def filesToDownload: Seq[String] = {
    WMT16Dataset.devArchives.map(archive => s"${WMT16Dataset.newsCommentaryUrl}/$archive.tgz") ++
        WMT16Dataset.testArchives.map(archive => s"${WMT16Dataset.newsCommentaryUrl}/$archive.tgz")
  }

  /** Returns all the train corpora (tuples containing name, source file, and target file) of this dataset. */
  override def trainCorpora: Seq[(String, File, File)] = {
    commonCrawlDataset.map(_.trainCorpora).getOrElse(Seq.empty) ++
        europarlV7Dataset.map(_.trainCorpora).getOrElse(Seq.empty) ++
        europarlV8Dataset.map(_.trainCorpora).getOrElse(Seq.empty) ++
        newsCommentaryV11Dataset.map(_.trainCorpora).getOrElse(Seq.empty)
  }

  /** Returns all the dev corpora (tuples containing name, source file, and target file) of this dataset. */
  override def devCorpora: Seq[(String, File, File)] = {
    WMT16Dataset.devArchives.foreach(archive => {
      (File(downloadsDir) / archive)
          .copyTo(File(downloadsDir) / "dev", overwrite = true)
    })
    var corpora = Seq.empty[(String, File, File)]
    val supported2008Languages = Set[Language](Czech, English, French, German, Hungarian, Spanish)
    if (supported2008Languages.contains(srcLanguage) && supported2008Languages.contains(tgtLanguage))
      corpora ++= Seq(("WMT16/newstest2008",
          File(downloadsDir) / "dev" / "dev" / s"news-test2008-src.$src",
          File(downloadsDir) / "dev" / "dev" / s"news-test2008-ref.$tgt"))
    val supported2009Languages = Set[Language](Czech, English, French, German, Hungarian, Italian, Spanish)
    if (supported2009Languages.contains(srcLanguage) && supported2009Languages.contains(tgtLanguage))
      corpora ++= Seq(("WMT16/newstest2009",
          File(downloadsDir) / "dev" / "dev" / s"newstest2009-src.$src",
          File(downloadsDir) / "dev" / "dev" / s"newstest2009-ref.$tgt"))
    val supported2009SysCombLanguages = Set[Language](Czech, English, French, German, Hungarian, Italian, Spanish)
    if (supported2009SysCombLanguages.contains(srcLanguage) && supported2009SysCombLanguages.contains(tgtLanguage))
      corpora ++= Seq(("WMT16/newssyscomb2009",
          File(downloadsDir) / "dev" / "dev" / s"newssyscomb2009-src.$src",
          File(downloadsDir) / "dev" / "dev" / s"newssyscomb2009-ref.$tgt"))
    val supported2010Languages = Set[Language](Czech, English, French, German, Spanish)
    if (supported2010Languages.contains(srcLanguage) && supported2010Languages.contains(tgtLanguage))
      corpora ++= Seq(("WMT16/newstest2010",
          File(downloadsDir) / "dev" / "dev" / s"newstest2010-src.$src",
          File(downloadsDir) / "dev" / "dev" / s"newstest2010-ref.$tgt"))
    val supported2011Languages = Set[Language](Czech, English, French, German, Spanish)
    if (supported2011Languages.contains(srcLanguage) && supported2011Languages.contains(tgtLanguage))
      corpora ++= Seq(("WMT16/newstest2011",
          File(downloadsDir) / "dev" / "dev" / s"newstest2011-src.$src",
          File(downloadsDir) / "dev" / "dev" / s"newstest2011-ref.$tgt"))
    val supported2012Languages = Set[Language](Czech, English, French, German, Russian, Spanish)
    if (supported2012Languages.contains(srcLanguage) && supported2012Languages.contains(tgtLanguage))
      corpora ++= Seq(("WMT16/newstest2012",
          File(downloadsDir) / "dev" / "dev" / s"newstest2012-src.$src",
          File(downloadsDir) / "dev" / "dev" / s"newstest2012-ref.$tgt"))
    val supported2013Languages = Set[Language](Czech, English, French, German, Russian, Spanish)
    if (supported2013Languages.contains(srcLanguage) && supported2013Languages.contains(tgtLanguage))
      corpora ++= Seq(("WMT16/newstest2013",
          File(downloadsDir) / "dev" / "dev" / s"newstest2013-src.$src",
          File(downloadsDir) / "dev" / "dev" / s"newstest2013-ref.$tgt"))
    val pair = if (reversed) s"$tgt$src" else s"$src$tgt"
    val supported2014Languages = Set[Language](Czech, English, French, German, Hindi, Russian)
    if (supported2014Languages.contains(srcLanguage) && supported2014Languages.contains(tgtLanguage))
      corpora ++= Seq(("WMT16/newstest2014",
          File(downloadsDir) / "dev" / "dev" / s"newstest2014-$pair-src.$src",
          File(downloadsDir) / "dev" / "dev" / s"newstest2014-$pair-ref.$tgt"))
    val supported2014DevLanguages = Set[Language](English, Hindi)
    if (supported2014DevLanguages.contains(srcLanguage) && supported2014DevLanguages.contains(tgtLanguage))
      corpora ++= Seq(("WMT16/newsdev2014",
          File(downloadsDir) / "dev" / "dev" / s"newsdev2014-src.$src",
          File(downloadsDir) / "dev" / "dev" / s"newsdev2014-ref.$tgt"))
    val supported2015Languages = Set[Language](Czech, English, Finnish, German, Russian)
    if (supported2015Languages.contains(srcLanguage) && supported2015Languages.contains(tgtLanguage))
      corpora ++= Seq(("WMT16/newstest2015",
          File(downloadsDir) / "dev" / "dev" / s"newstest2015-$src$tgt-src.$src",
          File(downloadsDir) / "dev" / "dev" / s"newstest2015-$src$tgt-ref.$tgt"))
    val supported2015DevLanguages = Set[Language](English, Finnish)
    if (supported2015DevLanguages.contains(srcLanguage) && supported2015DevLanguages.contains(tgtLanguage))
      corpora ++= Seq(("WMT16/newsdev2015",
          File(downloadsDir) / "dev" / "dev" / s"newsdev2015-$src$tgt-src.$src",
          File(downloadsDir) / "dev" / "dev" / s"newsdev2015-$src$tgt-ref.$tgt"))
    val supported2015DiscussDevLanguages = Set[Language](English, French)
    if (supported2015DiscussDevLanguages.contains(srcLanguage) &&
        supported2015DiscussDevLanguages.contains(tgtLanguage))
      corpora ++= Seq(("WMT16/newsdiscussdev2015",
          File(downloadsDir) / "dev" / "dev" / s"newsdiscussdev2015-$src$tgt-src.$src",
          File(downloadsDir) / "dev" / "dev" / s"newsdiscussdev2015-$src$tgt-ref.$tgt"))
    val supported2015DiscussTestLanguages = Set[Language](English, French)
    if (supported2015DiscussTestLanguages.contains(srcLanguage) &&
        supported2015DiscussTestLanguages.contains(tgtLanguage))
      corpora ++= Seq(("WMT16/newsdiscusstest2015",
          File(downloadsDir) / "dev" / "dev" / s"newsdiscusstest2015-$src$tgt-src.$src",
          File(downloadsDir) / "dev" / "dev" / s"newsdiscusstest2015-$src$tgt-ref.$tgt"))
    val supported2016DevLanguages = Set[Language](English, Romanian, Turkish)
    if (supported2016DevLanguages.contains(srcLanguage) && supported2016DevLanguages.contains(tgtLanguage))
      corpora ++= Seq(("WMT16/newsdev2016",
          File(downloadsDir) / "dev" / "dev" / s"newsdev2016-$src$tgt-src.$src",
          File(downloadsDir) / "dev" / "dev" / s"newsdev2016-$src$tgt-ref.$tgt"))
    corpora
  }

  /** Returns all the test corpora (tuples containing name, source file, and target file) of this dataset. */
  override def testCorpora: Seq[(String, File, File)] = {
    WMT16Dataset.testArchives.foreach(archive => {
      (File(downloadsDir) / archive)
          .copyTo(File(downloadsDir) / "test", overwrite = true)
    })
    var corpora = Seq.empty[(String, File, File)]
    val supported2016Languages = Set[Language](Czech, English, Finnish, German, Romanian, Russian, Turkish)
    if (supported2016Languages.contains(srcLanguage) && supported2016Languages.contains(tgtLanguage))
      corpora ++= Seq(("WMT16/newstest2016",
          File(downloadsDir) / "test" / "test" / s"newstest2016-$src$tgt-src.$src",
          File(downloadsDir) / "test" / "test" / s"newstest2016-$src$tgt-ref.$tgt"))
    corpora
  }
}

object WMT16Dataset {
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
      workingDir: Path,
      srcLanguage: Language,
      tgtLanguage: Language,
      bufferSize: Int = 8192,
      tokenize: Boolean = true,
      trainDataSentenceLengthBounds: (Int, Int) = null
  ): WMT16Dataset = {
    new WMT16Dataset(workingDir, srcLanguage, tgtLanguage, bufferSize, tokenize, trainDataSentenceLengthBounds)
  }
}
