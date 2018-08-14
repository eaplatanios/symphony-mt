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

package org.platanios.symphony.mt

import org.platanios.tensorflow.api._

/**
  * @author Emmanouil Antonios Platanios
  */
package object models {
  /** Contains the source language, the target language, a sentence batch, and the corresponding sentence lengths. */
  type TFBatchWithLanguages = (Output, Output, Output, Output)
  type TFBatchWithLanguagesT = (Tensor[DataType], Tensor[DataType], Tensor[DataType], Tensor[DataType])
  type TFBatchWithLanguagesD = (DataType, DataType, DataType, DataType)
  type TFBatchWithLanguagesS = (Shape, Shape, Shape, Shape)

  type TFBatchWithLanguage = (Output, Output, Output)
  type TFBatchWithLanguageT = (Tensor[DataType], Tensor[DataType], Tensor[DataType])
  type TFBatchWithLanguageD = (DataType, DataType, DataType)
  type TFBatchWithLanguageS = (Shape, Shape, Shape)

  type TFBatch = (Output, Output)
  type TFBatchT = (Tensor[DataType], Tensor[DataType])
  type TFBatchD = (DataType, DataType)
  type TFBatchS = (Shape, Shape)

  type TFLanguagePair = (Output, Output)
  type TFLanguagePairT = (Tensor[DataType], Tensor[DataType])
  type TFLanguagePairD = (DataType, DataType)
  type TFLanguagePairS = (Shape, Shape)

  type TFSentencesDataset = tf.data.Dataset[TFBatchT, TFBatch, TFBatchD, TFBatchS]

  type TFSentencePairsDataset = tf.data.Dataset[
      (TFLanguagePairT, (TFBatchT, TFBatchT)),
      (TFLanguagePair, (TFBatch, TFBatch)),
      (TFLanguagePairD, (TFBatchD, TFBatchD)),
      (TFLanguagePairS, (TFBatchS, TFBatchS))]

  type TFInputDataset = tf.data.Dataset[
      TFBatchWithLanguagesT, TFBatchWithLanguages,
      TFBatchWithLanguagesD, TFBatchWithLanguagesS]

  type TFTrainDataset = tf.data.Dataset[
      (TFBatchWithLanguagesT, TFBatchT),
      (TFBatchWithLanguages, TFBatch),
      (TFBatchWithLanguagesD, TFBatchD),
      (TFBatchWithLanguagesS, TFBatchS)]
}
