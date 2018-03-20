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
import org.platanios.symphony.mt.vocabulary.Vocabulary

import better.files.File

/**
  * @author Emmanouil Antonios Platanios
  */
class FileParallelDataset protected (
    override val name: String,
    override val vocabulary: Map[Language, Vocabulary],
    val dataConfig: DataConfig,
    val files: Map[Language, Seq[File]],
    val fileTypes: Seq[DatasetType] = null,
    val fileTags: Seq[ParallelDataset.Tag] = null
) extends ParallelDataset {
  override def isEmpty: Boolean = files.head._2.isEmpty
  override def nonEmpty: Boolean = !isEmpty

  override def filterLanguages(languages: Language*): FileParallelDataset = {
    FileParallelDataset(
      s"$name/${languages.map(_.abbreviation).mkString("-")}",
      vocabulary.filterKeys(languages.contains), dataConfig,
      files.filterKeys(languages.contains), fileTypes, fileTags)
  }

  override def filterTypes(types: DatasetType*): FileParallelDataset = {
    val filteredGroupedFiles = files.mapValues(_.zip(fileTypes).filter(f => types.contains(f._2)).map(_._1))
    val filteredFileTypes = fileTypes.filter(types.contains)
    val filteredFileKeys = fileTags.zip(fileTypes).filter(f => types.contains(f._2)).map(_._1)
    FileParallelDataset(
      s"$name/${types.mkString("-")}", vocabulary, dataConfig,
      filteredGroupedFiles, filteredFileTypes, filteredFileKeys)
  }

  override def filterTags(tags: ParallelDataset.Tag*): FileParallelDataset = {
    require(fileTags.nonEmpty, "Cannot filter a parallel dataset by file key when it contains no file keys.")
    val filteredGroupedFiles = files.mapValues(_.zip(fileTags).filter(f => tags.contains(f._2)).map(_._1))
    val filteredFileTypes = fileTags.zip(fileTypes).filter(f => tags.contains(f._1)).map(_._2)
    val filteredFileKeys = fileTags.filter(tags.contains)
    FileParallelDataset(
      s"$name/${tags.mkString("-")}", vocabulary, dataConfig,
      filteredGroupedFiles, filteredFileTypes, filteredFileKeys)
  }
}

object FileParallelDataset {
  def apply(
      name: String,
      vocabularies: Map[Language, Vocabulary],
      dataConfig: DataConfig,
      files: Map[Language, Seq[File]],
      fileTypes: Seq[DatasetType] = null,
      fileTags: Seq[ParallelDataset.Tag] = null
  ): FileParallelDataset = {
    new FileParallelDataset(name, vocabularies, dataConfig, files, fileTypes, fileTags)
  }
}
