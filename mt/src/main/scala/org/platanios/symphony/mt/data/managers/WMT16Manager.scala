/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

/**
  * @author Emmanouil Antonios Platanios
  */
case class WMT16Manager(srcLanguage: Language, tgtLanguage: Language) {
  require(
    WMT16Manager.supportedLanguagePairs.contains((srcLanguage, tgtLanguage)) ||
        WMT16Manager.supportedLanguagePairs.contains((tgtLanguage, srcLanguage)),
    "The provided language pair is not supported by the WMT16 data manager.")

  val src: String = srcLanguage.abbreviation
  val tgt: String = tgtLanguage.abbreviation

  val name: String = s"$src-$tgt"

  private[this] val reversed: Boolean = WMT16Manager.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))

  private[this] val corpusFilenamePrefix: String = {
    s"news-commentary-v11.${if (reversed) s"$tgt-$src" else s"$src-$tgt"}"
  }

  def download(path: Path, bufferSize: Int = 8192): ParallelDataset = {
    // Download and decompress the data, if necessary.
    val trainPath = path.resolve("train")
    val devPath = path.resolve("dev")
    val testPath = path.resolve("test")

    val newsCommentaryV11Manager = NewsCommentaryV11Manager(srcLanguage, tgtLanguage)
    val newsCommentaryV11Dataset = newsCommentaryV11Manager.download(path, bufferSize)

    downloadUpdatedArchives("dev", devPath, WMT16Manager.devArchives, bufferSize)
    downloadUpdatedArchives("test", testPath, WMT16Manager.testArchives, bufferSize)

    // Clone the Moses repository, if necessary.
    val mosesDecoder = Utilities.MosesDecoder(path.resolve("moses"))
    if (!mosesDecoder.exists)
      mosesDecoder.cloneRepository()

    // Convert the SGM files to simple text files.
    val srcDevCorporaSGM = Seq(
      devPath.resolve(s"newstest2014-$src$tgt-src.$src.sgm"),
      devPath.resolve(s"newstest2015-$src$tgt-src.$src.sgm"))
    val srcDevCorpora = Seq(
      devPath.resolve(s"newstest2014-$src$tgt-src.$src"),
      devPath.resolve(s"newstest2015-$src$tgt-src.$src"))
    val tgtDevCorporaSGM = Seq(
      devPath.resolve(s"newstest2014-$src$tgt-src.$tgt.sgm"),
      devPath.resolve(s"newstest2015-$src$tgt-src.$tgt.sgm"))
    val tgtDevCorpora = Seq(
      devPath.resolve(s"newstest2014-$src$tgt-src.$tgt"),
      devPath.resolve(s"newstest2015-$src$tgt-src.$tgt"))
    val srcTestCorporaSGM = Seq(testPath.resolve(s"newstest2016-$src$tgt-src.$src.sgm"))
    val srcTestCorpora = Seq(testPath.resolve(s"newstest2016-$src$tgt-src.$src"))
    val tgtTestCorporaSGM = Seq(testPath.resolve(s"newstest2016-$src$tgt-src.$tgt.sgm"))
    val tgtTestCorpora = Seq(testPath.resolve(s"newstest2016-$src$tgt-src.$tgt"))
    srcDevCorporaSGM.zip(srcDevCorpora).foreach(p => mosesDecoder.sgmToText(p._1, p._2))
    tgtDevCorporaSGM.zip(tgtDevCorpora).foreach(p => mosesDecoder.sgmToText(p._1, p._2))
    srcTestCorporaSGM.zip(srcTestCorpora).foreach(p => mosesDecoder.sgmToText(p._1, p._2))
    tgtTestCorporaSGM.zip(tgtTestCorpora).foreach(p => mosesDecoder.sgmToText(p._1, p._2))

    ParallelDataset(
      trainCorpora = newsCommentaryV11Dataset.trainCorpora,
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
}
