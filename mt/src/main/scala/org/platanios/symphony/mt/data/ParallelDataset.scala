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

package org.platanios.symphony.mt.data

import org.platanios.symphony.mt.Language

import java.nio.file.{Files, Path}

import scala.sys.process._

/**
  * @author Emmanouil Antonios Platanios
  */
case class ParallelDataset(languages: Seq[Language])(
    val trainCorpora: Seq[Seq[Path]] = Seq.fill(languages.size)(Seq.empty),
    val devCorpora: Seq[Seq[Path]] = Seq.fill(languages.size)(Seq.empty),
    val testCorpora: Seq[Seq[Path]] = Seq.fill(languages.size)(Seq.empty),
    val vocabularies: Seq[Seq[Path]] = Seq.fill(languages.size)(Seq.empty)) {
  private[this] def defaultPaths(prefix: String): Seq[Path] = {
    languages.zip(trainCorpora).map(p => p._2.headOption.map(_.resolveSibling(s"$prefix.${p._1.abbreviation}")).orNull)
  }

  def join(
      trainPaths: Seq[Path] = defaultPaths("train"),
      devPaths: Seq[Path] = defaultPaths("dev"),
      testPaths: Seq[Path] = defaultPaths("test"),
      vocabularyPaths: Seq[Path] = defaultPaths("vocab")
  ): ParallelDataset = {
    ParallelDataset(languages)(
      trainCorpora.zip(trainPaths).map(p => Seq(ParallelDataset.joinDatasets(p._1, p._2))),
      devCorpora.zip(devPaths).map(p => Seq(ParallelDataset.joinDatasets(p._1, p._2))),
      testCorpora.zip(testPaths).map(p => Seq(ParallelDataset.joinDatasets(p._1, p._2))),
      vocabularies.zip(vocabularyPaths).map(p => Seq(ParallelDataset.joinDatasets(p._1, p._2))))
  }
}

object ParallelDataset {
  private[ParallelDataset] def joinDatasets(paths: Seq[Path], joinedPath: Path): Path = {
    if (!Files.exists(joinedPath))
      (("cat" +: paths.map(_.toAbsolutePath.toString)) #> joinedPath.toFile).!
    joinedPath
  }
}
