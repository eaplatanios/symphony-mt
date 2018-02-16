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
    val tensors: Map[Language, Seq[(Tensor, Tensor)]],
    val tensorTypes: Seq[DatasetType] = null,
    val tensorKeys: Seq[String] = null
) extends ParallelDataset {
  override def filterLanguages(languages: Language*): ParallelDataset = {
    languages.foreach(checkSupportsLanguage)
    TensorParallelDataset(
      s"$name/${languages.map(_.abbreviation).mkString("-")}",
      vocabulary.filterKeys(languages.contains), tensors.filterKeys(languages.contains),
      tensorTypes, tensorKeys)
  }

  override def filterTypes(types: DatasetType*): ParallelDataset = {
    require(tensorTypes.nonEmpty, "Cannot filter a parallel dataset by tensor type when it contains no tensor types.")
    val filteredGroupedTensors = tensors.mapValues(_.zip(tensorTypes).filter(f => types.contains(f._2)).map(_._1))
    val filteredFileTypes = tensorTypes.filter(types.contains)
    val filteredFileKeys = tensorKeys.zip(tensorTypes).filter(f => types.contains(f._2)).map(_._1)
    TensorParallelDataset(
      s"$name/${types.mkString("-")}", vocabulary, filteredGroupedTensors,
      filteredFileTypes, filteredFileKeys)
  }

  override def filterKeys(keys: String*): ParallelDataset = {
    require(tensorKeys.nonEmpty, "Cannot filter a parallel dataset by tensor key when it contains no tensor keys.")
    val filteredGroupedTensors = tensors.mapValues(_.zip(tensorKeys).filter(f => keys.contains(f._2)).map(_._1))
    val filteredTensorTypes = tensorKeys.zip(tensorTypes).filter(f => keys.contains(f._1)).map(_._2)
    val filteredTensorsKeys = tensorKeys.filter(keys.contains)
    TensorParallelDataset(
      s"$name/${keys.mkString("-")}", vocabulary, filteredGroupedTensors,
      filteredTensorTypes, filteredTensorsKeys)
  }

  /** Creates and returns a TensorFlow dataset, for the specified language.
    *
    * Each element of that dataset is a tuple containing:
    *   - `INT32` tensor containing the input sentence word IDs, with shape `[batchSize, maxSentenceLength]`.
    *   - `INT32` tensor containing the input sentence lengths, with shape `[batchSize]`.
    *
    * @param  language Language for which the TensorFlow dataset is constructed.
    * @return Created TensorFlow dataset.
    */
  override def toTFMonolingual(language: Language): TFMonolingualDataset = {
    checkSupportsLanguage(language)
    joinBilingualDatasets(tensors(language).map(tf.data.TensorSlicesDataset(_)))
  }

  override def toTFBilingual(
      language1: Language,
      language2: Language,
      repeat: Boolean = true,
      isEval: Boolean = false
  ): TFBilingualDataset = ???
}

object TensorParallelDataset {
  def apply(
      name: String,
      vocabularies: Map[Language, Vocabulary],
      tensors: Map[Language, Seq[(Tensor, Tensor)]],
      tensorTypes: Seq[DatasetType] = null,
      tensorKeys: Seq[String] = null
  ): TensorParallelDataset = {
    new TensorParallelDataset(name, vocabularies, tensors, tensorTypes, tensorKeys)
  }
}
