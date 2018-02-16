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

package org.platanios.symphony.mt.translators.actors

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.data.ParallelDataset

import akka.actor.ActorRef

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class TrainScheduler[T <: ParallelDataset[T]](
    protected val dataset: ParallelDataset[T],
    protected val agents: Map[Language, ActorRef]
) {
  /** Initializes this train scheduler. This method is always called by the translation system, in order to start
    * the training process. */
  def initialize(): Unit

  /** Responds to a translation agent's train response. This method is called by the translation system, whenever it
    * receives an agent train response message. */
  def onTrainResponse(agent: ActorRef): Unit
}
