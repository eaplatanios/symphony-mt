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

package org.platanios.symphony.mt.models.curriculum.difficulty

import org.platanios.symphony.mt.models.SentencePairs
import org.platanios.tensorflow.api._

/**
  * @author Emmanouil Antonios Platanios
  */
class LengthBasedDifficulty[T](
    val lengthSelector: LengthBasedDifficulty.LengthSelector
) extends Difficulty[SentencePairs[T]] {
  override def apply(sample: SentencePairs[T]): Output[Float] = {
    val length = lengthSelector(srcLength = sample._2._2, tgtLength = sample._3._2)
    val part1 = 0.0f                     // Length 1 to 10.
    val part2 = (length - 10.0f) / 30.0f // Length 10 to 10.
    val part3 = 0.5f + length / 100.0f   // Length 20 to 100.
    tf.minimum(tf.maximum(part1, tf.minimum(part2, part3)), 1.0f)
  }
}

object LengthBasedDifficulty {
  trait LengthSelector {
    def apply(srcLength: Output[Int], tgtLength: Output[Int]): Output[Float]
  }

  object SourceLengthSelector extends LengthSelector {
    override def apply(srcLength: Output[Int], tgtLength: Output[Int]): Output[Float] = {
      srcLength.toFloat
    }
  }

  object TargetLengthSelector extends LengthSelector {
    override def apply(srcLength: Output[Int], tgtLength: Output[Int]): Output[Float] = {
      tgtLength.toFloat
    }
  }

  object MeanLengthSelector extends LengthSelector {
    override def apply(srcLength: Output[Int], tgtLength: Output[Int]): Output[Float] = {
      (srcLength.toFloat + tgtLength.toFloat) / 2.0f
    }
  }
}
