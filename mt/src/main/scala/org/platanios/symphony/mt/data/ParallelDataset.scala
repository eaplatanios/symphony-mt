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

package org.platanios.symphony.mt.data

import org.platanios.symphony.mt.Language

import java.nio.file.{Files, Path}

import scala.sys.process._

/**
  * @author Emmanouil Antonios Platanios
  */
case class ParallelDataset(
    workingDir: Path,
    private[data] val trainCorpora: Map[Language, Seq[Path]] = Map.empty,
    private[data] val devCorpora: Map[Language, Seq[Path]] = Map.empty,
    private[data] val testCorpora: Map[Language, Seq[Path]] = Map.empty,
    private[data] val vocabularies: Map[Language, Seq[Path]] = Map.empty
) {
  def trainCorpus(path: Path = workingDir): Map[Language, Path] = trainCorpora.map {
    case (language, paths) =>
      (language, ParallelDataset.joinDatasets(paths, path.resolve(s"train.${language.abbreviation}")))
  }

  def devCorpus(path: Path = workingDir): Map[Language, Path] = devCorpora.map {
    case (language, paths) =>
      (language, ParallelDataset.joinDatasets(paths, path.resolve(s"dev.${language.abbreviation}")))
  }

  def testCorpus(path: Path = workingDir): Map[Language, Path] = testCorpora.map {
    case (language, paths) =>
      (language, ParallelDataset.joinDatasets(paths, path.resolve(s"test.${language.abbreviation}")))
  }

  def vocabulary(path: Path = workingDir): Map[Language, Path] = vocabularies.map {
    case (language, paths) =>
      (language, ParallelDataset.joinDatasets(paths, path.resolve(s"vocab.${language.abbreviation}")))
  }
}

object ParallelDataset {
  private[ParallelDataset] def joinDatasets(paths: Seq[Path], joinedPath: Path): Path = {
    if (!Files.exists(joinedPath))
      (("cat" +: paths.map(_.toAbsolutePath.toString)) #> joinedPath.toFile).!
    joinedPath
  }
}
