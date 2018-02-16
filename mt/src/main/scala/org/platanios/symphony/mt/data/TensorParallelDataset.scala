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
    val tensorTypes: Seq[DatasetType],
    override val dataConfig: DataConfig,
    val tensorKeys: Seq[String] = null
) extends ParallelDataset[TensorParallelDataset] {
  override def filterLanguages(languages: Language*): TensorParallelDataset = {
    languages.foreach(checkSupportsLanguage)
    TensorParallelDataset(
      s"$name/${languages.map(_.abbreviation).mkString("-")}",
      vocabulary.filterKeys(languages.contains), tensors.filterKeys(languages.contains),
      tensorTypes, dataConfig)
  }

  override def filterTypes(types: DatasetType*): TensorParallelDataset = {
    val filteredGroupedTensors = tensors.mapValues(_.zip(tensorTypes).filter(f => types.contains(f._2)).map(_._1))
    val filteredFileTypes = tensorTypes.filter(types.contains)
    val filteredFileKeys = tensorKeys.zip(tensorTypes).filter(f => types.contains(f._2)).map(_._1)
    TensorParallelDataset(
      s"$name/${types.mkString("-")}", vocabulary, filteredGroupedTensors,
      filteredFileTypes, dataConfig, filteredFileKeys)
  }

  override def filterKeys(keys: String*): TensorParallelDataset = {
    require(tensorKeys.nonEmpty, "Cannot filter a parallel dataset by tensor key when it contains no tensor keys.")
    val filteredGroupedTensors = tensors.mapValues(_.zip(tensorKeys).filter(f => keys.contains(f._2)).map(_._1))
    val filteredTensorTypes = tensorKeys.zip(tensorTypes).filter(f => keys.contains(f._1)).map(_._2)
    val filteredTensorsKeys = tensorKeys.filter(keys.contains)
    TensorParallelDataset(
      s"$name/${keys.mkString("-")}", vocabulary, filteredGroupedTensors,
      filteredTensorTypes, dataConfig, filteredTensorsKeys)
  }

  /** Creates and returns a TensorFlow dataset, for the specified language.
    *
    * Each element of that dataset is a tuple containing:
    *   - `INT32` tensor containing the input sentence word IDs, with shape `[batchSize, maxSentenceLength]`.
    *   - `INT32` tensor containing the input sentence lengths, with shape `[batchSize]`.
    *
    * @param  language   Language for which the TensorFlow dataset is constructed.
    * @param  dataConfig Data configuration to use (optionally overriding this dataset's configuration).
    * @return Created TensorFlow dataset.
    */
  override def toTFMonolingual(language: Language, dataConfig: DataConfig = dataConfig): TFMonolingualDataset = {
    checkSupportsLanguage(language)

    val batchSize = dataConfig.inferBatchSize
    val vocabTable = vocabulary(language).lookupTable()
    val srcEosId = vocabTable.lookup(tf.constant(dataConfig.endOfSequenceToken)).cast(INT32)

    val batchingFn = (dataset: TFMonolingualDataset) => {
      dataset.dynamicPaddedBatch(
        batchSize,
        // The first entry represents the source line rows, which are unknown-length vectors.
        // The last entry is the source row size, which is a scalar.
        (Shape(-1), Shape.scalar()),
        // We pad the source sequences with 'endSequenceToken' tokens. Though notice that we do
        // not generally need to do this since later on we will be masking out calculations past
        // the true sequence.
        (srcEosId, tf.zeros(INT32, Shape.scalar())))
    }

    val datasetJoined = joinBilingualDatasets(tensors(language).map(tf.data.TensorSlicesDataset(_)))
    val datasetBeforeBatching = datasetJoined
    // TODO: Enforce length checks and cropping.
    // TODO: Check if the tensor is string-valued.

    batchingFn(datasetBeforeBatching)
  }

  override def toTFBilingual(
      language1: Language,
      language2: Language,
      dataConfig: DataConfig = dataConfig,
      repeat: Boolean = true,
      isEval: Boolean = false
  ): TFBilingualDataset = ???
}

object TensorParallelDataset {
  def apply(
      name: String,
      vocabularies: Map[Language, Vocabulary],
      groupedTensors: Map[Language, Seq[(Tensor, Tensor)]],
      tensorTypes: Seq[DatasetType],
      dataConfig: DataConfig,
      tensorKeys: Seq[String] = null
  ): TensorParallelDataset = {
    new TensorParallelDataset(name, vocabularies, groupedTensors, tensorTypes, dataConfig, tensorKeys)
  }
}
