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

package org.platanios.symphony.mt.data.managers

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.Language._
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.data.utilities.CompressedFiles

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.{Files, Path}

import scala.collection.mutable.ListBuffer

// TODO: Make sure to avoid re-doing work if this path exists and contains the joined dataset.

/**
  *
  * More information on this dataset can be found [here](http://www.statmt.org/wmt16/translation-task.html).
  *
  * @author Emmanouil Antonios Platanios
  */
case class WMT16Manager(
    workingDir: Path,
    srcLanguage: Language,
    tgtLanguage: Language
) extends Manager(workingDir.resolve("wmt-16")) {
  require(
    WMT16Manager.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the WMT16 data manager.")

  val src: String = srcLanguage.abbreviation
  val tgt: String = tgtLanguage.abbreviation

  val name: String = s"$src-$tgt"

  private[this] val reversed: Boolean = {
    WMT16Manager.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  override def download(bufferSize: Int = 8192): ParallelDataset = {
    // Download and decompress the train data, if necessary.
    val srcTrainCorpora = ListBuffer.empty[Path]
    val tgtTrainCorpora = ListBuffer.empty[Path]

    if (EuroparlV7Manager.isLanguagePairSupported(srcLanguage, tgtLanguage)) {
      val europarlV7Manager = EuroparlV7Manager(workingDir, srcLanguage, tgtLanguage)
      val europarlV7Dataset = europarlV7Manager.download(bufferSize)
      srcTrainCorpora ++= europarlV7Dataset.trainCorpora(srcLanguage)
      tgtTrainCorpora ++= europarlV7Dataset.trainCorpora(tgtLanguage)
    }

    if (EuroparlV8Manager.isLanguagePairSupported(srcLanguage, tgtLanguage)) {
      val europarlV8Manager = EuroparlV8Manager(workingDir, srcLanguage, tgtLanguage)
      val europarlV8Dataset = europarlV8Manager.download(bufferSize)
      srcTrainCorpora ++= europarlV8Dataset.trainCorpora(srcLanguage)
      tgtTrainCorpora ++= europarlV8Dataset.trainCorpora(tgtLanguage)
    }

    if (CommonCrawlManager.isLanguagePairSupported(srcLanguage, tgtLanguage)) {
      val commonCrawlManager = CommonCrawlManager(workingDir, srcLanguage, tgtLanguage)
      val commonCrawlDataset = commonCrawlManager.download(bufferSize)
      srcTrainCorpora ++= commonCrawlDataset.trainCorpora(srcLanguage)
      tgtTrainCorpora ++= commonCrawlDataset.trainCorpora(tgtLanguage)
    }

    if (NewsCommentaryV11Manager.isLanguagePairSupported(srcLanguage, tgtLanguage)) {
      val newsCommentaryV11Manager = NewsCommentaryV11Manager(workingDir, srcLanguage, tgtLanguage)
      val newsCommentaryV11Dataset = newsCommentaryV11Manager.download(bufferSize)
      srcTrainCorpora ++= newsCommentaryV11Dataset.trainCorpora(srcLanguage)
      tgtTrainCorpora ++= newsCommentaryV11Dataset.trainCorpora(tgtLanguage)
    }

    // TODO: Add support for the "CzEng 1.6pre" dataset.
    // TODO: Add support for the "Yandex Corpus" dataset.
    // TODO: Add support for the "Wiki Headlines" dataset.
    // TODO: Add support for the "SETIMES2" dataset.

    // Clone the Moses repository, if necessary.
    val mosesDecoder = Utilities.MosesDecoder(path.resolve("moses"))
    if (!mosesDecoder.exists)
      mosesDecoder.cloneRepository()

    // Download, decompress, and collect the dev and the test files, if necessary.
    downloadUpdatedArchives("dev", path, WMT16Manager.devArchives, bufferSize)
    downloadUpdatedArchives("test", path, WMT16Manager.testArchives, bufferSize)
    val devPath = path.resolve("dev").resolve("dev")
    val testPath = path.resolve("test").resolve("test")
    val (srcDevCorpora, tgtDevCorpora) = collectDevCorpora(devPath, mosesDecoder)
    val (srcTestCorpora, tgtTestCorpora) = collectTestCorpora(testPath, mosesDecoder)

    ParallelDataset(
      workingDir = path,
      trainCorpora = Map(srcLanguage -> srcTrainCorpora, tgtLanguage -> tgtTrainCorpora),
      devCorpora = Map(srcLanguage -> srcDevCorpora, tgtLanguage -> tgtDevCorpora),
      testCorpora = Map(srcLanguage -> srcTestCorpora, tgtLanguage -> tgtTestCorpora))
  }

  protected def downloadUpdatedArchives(
      archiveName: String, path: Path, archives: Seq[String], bufferSize: Int = 8192
  ): Unit = {
    val archivePath = path.resolve(archiveName)
    if (!Files.exists(archivePath)) {
      archives.foreach(archive => {
        val currentPath = path.resolve(s"$archive.tgz")
        Manager.maybeDownload(currentPath, s"${WMT16Manager.newsCommentaryUrl}/$archive.tgz", bufferSize)
        CompressedFiles.decompressTGZ(currentPath, archivePath)
      })
    }
  }

  protected def collectDevCorpora(path: Path, mosesDecoder: Utilities.MosesDecoder): (Seq[Path], Seq[Path]) = {
    var srcPaths = ListBuffer.empty[Path]
    var tgtPaths = ListBuffer.empty[Path]

    // 2008 Data
    val supported2008Languages = Set[Language](Czech, English, French, German, Hungarian, Spanish)
    if (supported2008Languages.contains(srcLanguage) && supported2008Languages.contains(tgtLanguage)) {
      val srcPath = path.resolve(s"news-test2008-src.$src")
      val tgtPath = path.resolve(s"news-test2008-ref.$tgt")
      mosesDecoder.sgmToText(path.resolve(s"news-test2008-src.$src.sgm"), srcPath)
      mosesDecoder.sgmToText(path.resolve(s"news-test2008-ref.$tgt.sgm"), tgtPath)
      srcPaths += srcPath
      tgtPaths += tgtPath
    }

    // 2009 Data
    val supported2009Languages = Set[Language](Czech, English, French, German, Hungarian, Italian, Spanish)
    if (supported2009Languages.contains(srcLanguage) && supported2009Languages.contains(tgtLanguage)) {
      val srcPath = path.resolve(s"newstest2009-src.$src")
      val tgtPath = path.resolve(s"newstest2009-ref.$tgt")
      mosesDecoder.sgmToText(path.resolve(s"newstest2009-src.$src.sgm"), srcPath)
      mosesDecoder.sgmToText(path.resolve(s"newstest2009-ref.$tgt.sgm"), tgtPath)
      srcPaths += srcPath
      tgtPaths += tgtPath
    }

    val supported2009SysCombLanguages = Set[Language](Czech, English, French, German, Hungarian, Italian, Spanish)
    if (supported2009SysCombLanguages.contains(srcLanguage) && supported2009SysCombLanguages.contains(tgtLanguage)) {
      val srcPath = path.resolve(s"newssyscomb2009-src.$src")
      val tgtPath = path.resolve(s"newssyscomb2009-ref.$tgt")
      mosesDecoder.sgmToText(path.resolve(s"newssyscomb2009-src.$src.sgm"), srcPath)
      mosesDecoder.sgmToText(path.resolve(s"newssyscomb2009-ref.$tgt.sgm"), tgtPath)
      srcPaths += srcPath
      tgtPaths += tgtPath
    }

    // 2010 Data
    val supported2010Languages = Set[Language](Czech, English, French, German, Spanish)
    if (supported2010Languages.contains(srcLanguage) && supported2010Languages.contains(tgtLanguage)) {
      val srcPath = path.resolve(s"newstest2010-src.$src")
      val tgtPath = path.resolve(s"newstest2010-ref.$tgt")
      mosesDecoder.sgmToText(path.resolve(s"newstest2010-src.$src.sgm"), srcPath)
      mosesDecoder.sgmToText(path.resolve(s"newstest2010-ref.$tgt.sgm"), tgtPath)
      srcPaths += srcPath
      tgtPaths += tgtPath
    }

    // 2011 Data
    val supported2011Languages = Set[Language](Czech, English, French, German, Spanish)
    if (supported2011Languages.contains(srcLanguage) && supported2011Languages.contains(tgtLanguage)) {
      val srcPath = path.resolve(s"newss"newstest2011-ref.$tgt"test2011-src.$src")
      val tgtPath = path.resolve()
      mosesDecoder.sgmToText(path.resolve(s"newstest2011-src.$src.sgm"), srcPath)
      mosesDecoder.sgmToText(path.resolve(s"newstest2011-ref.$tgt.sgm"), tgtPath)
      srcPaths += srcPath
      tgtPaths += tgtPath
    }

    // 2012 Data
    val supported2012Languages = Set[Language](Czech, English, French, German, Russian, Spanish)
    if (supported2012Languages.contains(srcLanguage) && supported2012Languages.contains(tgtLanguage)) {
      val srcPath = path.resolve(s"newstest2012-src.$src")
      val tgtPath = path.resolve(s"newstest2012-ref.$tgt")
      mosesDecoder.sgmToText(path.resolve(s"newstest2012-src.$src.sgm"), srcPath)
      mosesDecoder.sgmToText(path.resolve(s"newstest2012-ref.$tgt.sgm"), tgtPath)
      srcPaths += srcPath
      tgtPaths += tgtPath
    }

    // 2013 Data
    val supported2013Languages = Set[Language](Czech, English, French, German, Russian, Spanish)
    if (supported2013Languages.contains(srcLanguage) && supported2013Languages.contains(tgtLanguage)) {
      val srcPath = path.resolve(s"newstest2013-src.$src")
      val tgtPath = path.resolve(s"newstest2013-ref.$tgt")
      mosesDecoder.sgmToText(path.resolve(s"newstest2013-src.$src.sgm"), srcPath)
      mosesDecoder.sgmToText(path.resolve(s"newstest2013-ref.$tgt.sgm"), tgtPath)
      srcPaths += srcPath
      tgtPaths += tgtPath
    }

    // 2014 Data
    val supported2014Languages = Set[Language](Czech, English, French, German, Hindi, Russian)
    if (supported2014Languages.contains(srcLanguage) && supported2014Languages.contains(tgtLanguage)) {
      val pair = if (reversed) s"$tgt$src" else s"$src$tgt"
      val srcPath = path.resolve(s"newstest2014-$pair-src.$src")
      val tgtPath = path.resolve(s"newstest2014-$pair-ref.$tgt")
      mosesDecoder.sgmToText(path.resolve(s"newstest2014-$pair-src.$src.sgm"), srcPath)
      mosesDecoder.sgmToText(path.resolve(s"newstest2014-$pair-ref.$tgt.sgm"), tgtPath)
      srcPaths += srcPath
      tgtPaths += tgtPath
    }

    val supported2014DevLanguages = Set[Language](English, Hindi)
    if (supported2014DevLanguages.contains(srcLanguage) && supported2014DevLanguages.contains(tgtLanguage)) {
      val srcPath = path.resolve(s"newsdev2014-src.$src")
      val tgtPath = path.resolve(s"newsdev2014-ref.$tgt")
      mosesDecoder.sgmToText(path.resolve(s"newsdev2014-src.$src.sgm"), srcPath)
      mosesDecoder.sgmToText(path.resolve(s"newsdev2014-ref.$tgt.sgm"), tgtPath)
      srcPaths += srcPath
      tgtPaths += tgtPath
    }

    // 2015 Data
    val supported2015Languages = Set[Language](Czech, English, Finnish, German, Russian)
    if (supported2015Languages.contains(srcLanguage) && supported2015Languages.contains(tgtLanguage)) {
      val srcPath = path.resolve(s"newstest2015-$src$tgt-src.$src")
      val tgtPath = path.resolve(s"newstest2015-$src$tgt-ref.$tgt")
      mosesDecoder.sgmToText(path.resolve(s"newstest2015-$src$tgt-src.$src.sgm"), srcPath)
      mosesDecoder.sgmToText(path.resolve(s"newstest2015-$src$tgt-ref.$tgt.sgm"), tgtPath)
      srcPaths += srcPath
      tgtPaths += tgtPath
    }

    val supported2015DevLanguages = Set[Language](English, Finnish)
    if (supported2015DevLanguages.contains(srcLanguage) && supported2015DevLanguages.contains(tgtLanguage)) {
      val srcPath = path.resolve(s"newsdev2015-$src$tgt-src.$src")
      val tgtPath = path.resolve(s"newsdev2015-$src$tgt-ref.$tgt")
      mosesDecoder.sgmToText(path.resolve(s"newsdev2015-$src$tgt-src.$src.sgm"), srcPath)
      mosesDecoder.sgmToText(path.resolve(s"newsdev2015-$src$tgt-ref.$tgt.sgm"), tgtPath)
      srcPaths += srcPath
      tgtPaths += tgtPath
    }

    val supported2015DiscussDevLanguages = Set[Language](English, French)
    if (supported2015DiscussDevLanguages.contains(srcLanguage) &&
        supported2015DiscussDevLanguages.contains(tgtLanguage)) {
      val srcPath = path.resolve(s"newsdiscussdev2015-$src$tgt-src.$src")
      val tgtPath = path.resolve(s"newsdiscussdev2015-$src$tgt-ref.$tgt")
      mosesDecoder.sgmToText(path.resolve(s"newsdiscussdev2015-$src$tgt-src.$src.sgm"), srcPath)
      mosesDecoder.sgmToText(path.resolve(s"newsdiscussdev2015-$src$tgt-ref.$tgt.sgm"), tgtPath)
      srcPaths += srcPath
      tgtPaths += tgtPath
    }

    val supported2015DiscussTestLanguages = Set[Language](English, French)
    if (supported2015DiscussTestLanguages.contains(srcLanguage) &&
        supported2015DiscussTestLanguages.contains(tgtLanguage)) {
      val srcPath = path.resolve(s"newsdiscusstest2015-$src$tgt-src.$src")
      val tgtPath = path.resolve(s"newsdiscusstest2015-$src$tgt-ref.$tgt")
      mosesDecoder.sgmToText(path.resolve(s"newsdiscusstest2015-$src$tgt-src.$src.sgm"), srcPath)
      mosesDecoder.sgmToText(path.resolve(s"newsdiscusstest2015-$src$tgt-ref.$tgt.sgm"), tgtPath)
      srcPaths += srcPath
      tgtPaths += tgtPath
    }

    // 2016 Data
    val supported2016DevLanguages = Set[Language](English, Romanian, Turkish)
    if (supported2016DevLanguages.contains(srcLanguage) && supported2016DevLanguages.contains(tgtLanguage)) {
      val srcPath = path.resolve(s"newsdev2016-$src$tgt-src.$src")
      val tgtPath = path.resolve(s"newsdev2016-$src$tgt-ref.$tgt")
      mosesDecoder.sgmToText(path.resolve(s"newsdev2016-$src$tgt-src.$src.sgm"), srcPath)
      mosesDecoder.sgmToText(path.resolve(s"newsdev2016-$src$tgt-ref.$tgt.sgm"), tgtPath)
      srcPaths += srcPath
      tgtPaths += tgtPath
    }

    (srcPaths, tgtPaths)
  }

  protected def collectTestCorpora(path: Path, mosesDecoder: Utilities.MosesDecoder): (Seq[Path], Seq[Path]) = {
    var srcPaths = ListBuffer.empty[Path]
    var tgtPaths = ListBuffer.empty[Path]

    // 2016 Data
    val supported2016Languages = Set[Language](Czech, English, Finnish, German, Romanian, Russian, Turkish)
    if (supported2016Languages.contains(srcLanguage) && supported2016Languages.contains(tgtLanguage)) {
      val srcPath = path.resolve(s"newstest2016-$src$tgt-src.$src")
      val tgtPath = path.resolve(s"newstest2016-$src$tgt-ref.$tgt")
      mosesDecoder.sgmToText(path.resolve(s"newstest2016-$src$tgt-src.$src.sgm"), srcPath)
      mosesDecoder.sgmToText(path.resolve(s"newstest2016-$src$tgt-ref.$tgt.sgm"), tgtPath)
      srcPaths += srcPath
      tgtPaths += tgtPath
    }

    (srcPaths, tgtPaths)
  }
}

object WMT16Manager {
  private[WMT16Manager] val logger = Logger(LoggerFactory.getLogger("WMT16 Data Manager"))

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
}
