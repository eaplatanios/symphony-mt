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
class NewsCommentaryV11Loader(
    override val srcLanguage: Language,
    override val tgtLanguage: Language,
    val config: DataConfig
) extends ParallelDatasetLoader(srcLanguage = srcLanguage, tgtLanguage = tgtLanguage) {
  require(
    NewsCommentaryV11Loader.isLanguagePairSupported(srcLanguage, tgtLanguage),
    "The provided language pair is not supported by the News Commentary v11 dataset.")

  override def name: String = "News Commentary v11"

  override def dataConfig: DataConfig = {
    config.copy(dataDir =
        config.dataDir
            .resolve("news-commentary-v11")
            .resolve(s"${srcLanguage.abbreviation}-${tgtLanguage.abbreviation}"))
  }

  override def downloadsDir: Path = config.dataDir.resolve("news-commentary-v11").resolve("downloads")

  private[this] def reversed: Boolean = {
    NewsCommentaryV11Loader.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  private[this] def corpusFilenamePrefix: String = {
    s"news-commentary-v11.${if (reversed) s"$tgt-$src" else s"$src-$tgt"}"
  }

  /** Sequence of files to download as part of this dataset. */
  override def filesToDownload: Seq[(String, String)] = {
    Seq(
      (s"${NewsCommentaryV11Loader.url}/${NewsCommentaryV11Loader.archivePrefix}.tgz",
          s"${NewsCommentaryV11Loader.archivePrefix}.tgz"))
  }

  /** Returns all the corpora (tuples containing name, source file, target file, and a file processor to use)
    * of this dataset type. */
  override def corpora(datasetType: DatasetType): Seq[(ParallelDataset.Tag, File, File, FileProcessor)] = {
    datasetType match {
      case Train => Seq((NewsCommentaryV11Loader.Train,
          File(downloadsDir) / NewsCommentaryV11Loader.archivePrefix
              / NewsCommentaryV11Loader.archivePrefix / s"$corpusFilenamePrefix.$src",
          File(downloadsDir) / NewsCommentaryV11Loader.archivePrefix
              / NewsCommentaryV11Loader.archivePrefix / s"$corpusFilenamePrefix.$tgt",
          Normalizer))
      case _ => Seq.empty
    }
  }
}

object NewsCommentaryV11Loader {
  val url          : String = "http://data.statmt.org/wmt16/translation-task"
  val archivePrefix: String = "training-parallel-nc-v11"

  val supportedLanguagePairs: Set[(Language, Language)] = {
    Set((Czech, English), (German, English), (Russian, English))
  }

  def isLanguagePairSupported(srcLanguage: Language, tgtLanguage: Language): Boolean = {
    supportedLanguagePairs.contains((srcLanguage, tgtLanguage)) ||
        supportedLanguagePairs.contains((tgtLanguage, srcLanguage))
  }

  def apply(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataConfig: DataConfig
  ): NewsCommentaryV11Loader = {
    new NewsCommentaryV11Loader(srcLanguage, tgtLanguage, dataConfig)
  }

  case object Train extends ParallelDataset.Tag {
    override val value: String = "news-commentary-v11/train"
  }
}
