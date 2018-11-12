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

package org.platanios.symphony.mt.models

/**
  * @author Emmanouil Antonios Platanios
  */
trait Transformation[TrainIn, InferIn, TrainOut, InferOut] {
  def applyTrain(value: TrainIn)(implicit context: ModelConstructionContext): TrainOut
  def applyInfer(value: InferIn)(implicit context: ModelConstructionContext): InferOut

  def >>[ComposedTrainOut, ComposedInferOut](
      other: Transformation[TrainOut, InferOut, ComposedTrainOut, ComposedInferOut]
  ): Transformation[TrainIn, InferIn, ComposedTrainOut, ComposedInferOut] = {
    compose(other)
  }

  def compose[ComposedTrainOut, ComposedInferOut](
      other: Transformation[TrainOut, InferOut, ComposedTrainOut, ComposedInferOut]
  ): Transformation[TrainIn, InferIn, ComposedTrainOut, ComposedInferOut] = {
    val outerTransformation = this
    new Transformation[TrainIn, InferIn, ComposedTrainOut, ComposedInferOut] {
      override def applyTrain(source: TrainIn)(implicit context: ModelConstructionContext): ComposedTrainOut = {
        other.applyTrain(outerTransformation.applyTrain(source))
      }

      override def applyInfer(source: InferIn)(implicit context: ModelConstructionContext): ComposedInferOut = {
        other.applyInfer(outerTransformation.applyInfer(source))
      }
    }
  }
}

trait SimpleTransformation[In, Out] extends Transformation[In, In, Out, Out] {
  def apply(value: In)(implicit context: ModelConstructionContext): Out
  
  override def applyTrain(value: In)(implicit context: ModelConstructionContext): Out = {
    apply(value)
  }
  
  override def applyInfer(value: In)(implicit context: ModelConstructionContext): Out = {
    apply(value)
  }
}

object Transformation {
  trait Encoder[Code] extends SimpleTransformation[Sequences[Int], Code]
  trait Decoder[Code] extends Transformation[Code, Code, Sequences[Float], Sequences[Int]]
}
