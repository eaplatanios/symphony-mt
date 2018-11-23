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

package org.platanios.symphony.mt.models.decoders

import org.platanios.symphony.mt.models.ModelConstructionContext
import org.platanios.symphony.mt.models.helpers.Common
import org.platanios.tensorflow.api._

/**
  * @author Emmanouil Antonios Platanios
  */
trait OutputLayer {
  def apply[T: TF : IsNotQuantized](
      inputSize: Int
  )(implicit context: ModelConstructionContext): Output[T] => Output[T]
}

case object ProjectionToWords extends OutputLayer {
  override def apply[T: TF : IsNotQuantized](
      inputSize: Int
  )(implicit context: ModelConstructionContext): Output[T] => Output[T] = {
    logits: Output[T] => {
      val outputWeights = context.parameterManager.getProjectionToWords(inputSize, context.tgtLanguageID).castTo[T]
      Common.matrixMultiply(logits, outputWeights)
    }
  }
}

case object ProjectionToWordEmbeddings extends OutputLayer {
  override def apply[T: TF : IsNotQuantized](
      inputSize: Int
  )(implicit context: ModelConstructionContext): Output[T] => Output[T] = {
    logits: Output[T] => {
      val wordEmbeddingsSize = context.parameterManager.wordEmbeddingsType.embeddingsSize
      val outputWeights = context.parameterManager.get[T](
        "ProjectionToWordEmbeddings", Shape(inputSize, wordEmbeddingsSize))
      val projectedOutputs = Common.matrixMultiply(logits, outputWeights)
      val wordEmbeddings = context.parameterManager.wordEmbeddingsTable(context.tgtLanguageID).castTo[T]
      Common.matrixMultiply(projectedOutputs, wordEmbeddings, transposeY = true)
    }
  }
}
