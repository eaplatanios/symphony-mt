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
import org.platanios.tensorflow.api._

/**
  * @author Emmanouil Antonios Platanios
  */
class TensorParallelDataset protected (
    override val name: String,
    override val vocabulary: Map[Language, Vocabulary],
    val tensors: Map[Language, Seq[(Tensor[String], Tensor[Int])]],
    val tensorTypes: Seq[DatasetType] = null,
    val tensorTags: Seq[ParallelDataset.Tag] = null
) extends ParallelDataset {
  override def isEmpty: Boolean = tensors.head._2.isEmpty
  override def nonEmpty: Boolean = !isEmpty

  override def filterLanguages(languages: Language*): TensorParallelDataset = {
    TensorParallelDataset(
      s"$name/${languages.map(_.abbreviation).mkString("-")}",
      vocabulary.filterKeys(languages.contains), tensors.filterKeys(languages.contains),
      tensorTypes, tensorTags)
  }

  override def filterTypes(types: DatasetType*): TensorParallelDataset = {
    require(tensorTypes.nonEmpty, "Cannot filter a parallel dataset by tensor type when it contains no tensor types.")
    val filteredGroupedTensors = tensors.mapValues(_.zip(tensorTypes).filter(f => types.contains(f._2)).map(_._1))
    val filteredFileTypes = tensorTypes.filter(types.contains)
    val filteredFileKeys = tensorTags.zip(tensorTypes).filter(f => types.contains(f._2)).map(_._1)
    TensorParallelDataset(
      s"$name/${types.mkString("-")}", vocabulary, filteredGroupedTensors,
      filteredFileTypes, filteredFileKeys)
  }

  override def filterTags(tags: ParallelDataset.Tag*): TensorParallelDataset = {
    require(tensorTags.nonEmpty, "Cannot filter a parallel dataset by tensor key when it contains no tensor keys.")
    val filteredGroupedTensors = tensors.mapValues(_.zip(tensorTags).filter(f => tags.contains(f._2)).map(_._1))
    val filteredTensorTypes = tensorTags.zip(tensorTypes).filter(f => tags.contains(f._1)).map(_._2)
    val filteredTensorsKeys = tensorTags.filter(tags.contains)
    TensorParallelDataset(
      s"$name/${tags.mkString("-")}", vocabulary, filteredGroupedTensors,
      filteredTensorTypes, filteredTensorsKeys)
  }
}

object TensorParallelDataset {
  def apply(
      name: String,
      vocabularies: Map[Language, Vocabulary],
      tensors: Map[Language, Seq[(Tensor[String], Tensor[Int])]],
      tensorTypes: Seq[DatasetType] = null,
      tensorTags: Seq[ParallelDataset.Tag] = null
  ): TensorParallelDataset = {
    new TensorParallelDataset(name, vocabularies, tensors, tensorTypes, tensorTags)
  }
}
