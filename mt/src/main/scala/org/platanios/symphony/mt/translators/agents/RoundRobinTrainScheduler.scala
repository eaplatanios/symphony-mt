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

package org.platanios.symphony.mt.translators.agents

import akka.actor.ActorRef
import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.data.LoadedDataset
import org.platanios.symphony.mt.translators.agents.Messages.AgentTrainRequest

import java.util.concurrent.atomic.AtomicInteger

/**
  * @author Emmanouil Antonios Platanios
  */
class RoundRobinTrainScheduler protected (
    override protected val dataset: LoadedDataset,
    override protected val agents: Map[Language, ActorRef]
) extends TrainScheduler(dataset, agents) {
  /** Contains the train datasets used by this train scheduler. */
  protected val datasets: Map[Language, Seq[(Language, TrainScheduler.DatasetIterator)]] = {
    dataset.languagePairs.map(languages =>
      languages._1 -> (datasets.getOrElse(languages._1, Seq.empty) :+
          ((languages._2, TrainScheduler.DatasetIterator(
            dataset.files(languages._1, languages._2), dataset.dataConfig))))).toMap
  }

  protected var currentIndices: Map[ActorRef, (Language, AtomicInteger)] = {
    agents.map(pair => pair._2 -> ((pair._1, new AtomicInteger(0))))
  }

  /** Initializes this train scheduler. This method is always called by the translation system, in order to start
    * the training process. */
  override protected def initialize(): Unit = {
    agents.values.foreach(onTrainResponse)
  }

  /** Responds to a translation agent's train response. This method is called by the translation system, whenever it
    * receives an agent train response message. */
  override protected def onTrainResponse(agent: ActorRef): Unit = {
    val (lang, index) = currentIndices(agent)
    var nextIndex = index.incrementAndGet()
    val dataset = datasets(lang)
    if (nextIndex >= dataset.size) {
      index.set(0)
      nextIndex = 0
    }
    val nextDataset = dataset(nextIndex)
    agent ! AgentTrainRequest(agents(nextDataset._1), nextDataset._2.next())
  }
}

object RoundRobinTrainScheduler {
  def apply(dataset: LoadedDataset, agents: Map[Language, ActorRef]): RoundRobinTrainScheduler = {
    new RoundRobinTrainScheduler(dataset, agents)
  }
}
