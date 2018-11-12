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

package org.platanios.symphony.mt.models.curriculum

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Counter
import org.platanios.tensorflow.api.ops.FunctionGraph

/**
  * @author Emmanouil Antonios Platanios
  */
trait Curriculum[Sample] {
  protected def currentStep: Output[Long] = {
    // Obtain the outer graph (outside this function call) and get the global step variable defined in that graph.
    var graph = tf.currentGraph
    while (graph.isInstanceOf[FunctionGraph])
      graph = graph.asInstanceOf[FunctionGraph].outerGraph
    Counter.getOrCreate(Graph.Keys.GLOBAL_STEP, local = false, graph = graph).value
  }

  def samplesFilter: Option[Sample => Output[Boolean]] = {
    None
  }

  def >>(other: Curriculum[Sample]): Curriculum[Sample] = {
    compose(other)
  }

  def compose(other: Curriculum[Sample]): Curriculum[Sample] = {
    new Curriculum[Sample] {
      override def samplesFilter: Option[Sample => Output[Boolean]] = {
        (samplesFilter, other.samplesFilter) match {
          case (Some(thisFilter), Some(otherFilter)) =>
            Some((sample: Sample) => {
              tf.cond(
                thisFilter(sample),
                () => otherFilter(sample),
                () => tf.constant[Boolean](false))
            })
          case (Some(thisFilter), None) => Some(thisFilter)
          case (None, Some(otherFilter)) => Some(otherFilter)
          case (None, None) => None
        }
      }
    }
  }
}

object Curriculum {
  def none[T]: Curriculum[T] = {
    new Curriculum[T] {}
  }
}
