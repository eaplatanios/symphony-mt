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
import org.platanios.symphony.mt.data.{BilingualDatasetIterator, ParallelDataset, TensorParallelDataset}
import org.platanios.symphony.mt.translators.actors.Messages.{AgentSelfTrainRequest, AgentTrainRequest}
import org.platanios.tensorflow.api.learn.StopCriteria

import akka.actor._

import java.util.concurrent.atomic.AtomicInteger

/**
  * @author Emmanouil Antonios Platanios
  */
class RoundRobinTrainScheduler protected (
    override protected val dataset: ParallelDataset,
    override protected val agents: Map[Language, ActorRef],
    val selfTrainSteps: Long = 0L,
    val trainStepsPerRequest: Long = 10L
)(implicit sender: ActorRef = Actor.noSender) extends TrainScheduler(dataset, agents) {
  protected var completedSteps: Long = 0L

  /** Contains the train datasets used by this train scheduler. */
  protected val datasets: Map[Language, Seq[(Language, BilingualDatasetIterator)]] = {
    val aggregated = dataset.languagePairs(includeReversed = true).map {
      case (srcLang, tgtLang) =>
        srcLang -> ((tgtLang, BilingualDatasetIterator(dataset, srcLang, tgtLang, repeat = true, isEval = false)))
    }
    aggregated.toSeq.groupBy(_._1).mapValues(_.map(_._2))
  }

  protected var currentIndices: Map[ActorRef, (Language, AtomicInteger)] = {
    agents.map(pair => pair._2 -> ((pair._1, new AtomicInteger(0))))
  }

  /** Initializes this train scheduler. This method is always called by the translation system, in order to start
    * the training process. */
  override def initialize(): Unit = {
    agents.values.foreach(onTrainResponse)
  }

  /** Responds to a translation agent's train response. This method is called by the translation system, whenever it
    * receives an agent train response message. */
  override def onTrainResponse(agent: ActorRef): Unit = {
    completedSteps += 1L
    val (lang, index) = currentIndices(agent)
    var nextIndex = index.getAndIncrement()
    val dataset = datasets(lang)
    if (nextIndex >= dataset.size) {
      index.set(0)
      nextIndex = 0
    }
    val nextDataset = dataset(nextIndex)
    // TODO: !!! Make this more efficient. Creating new datasets can have an overhead.
    if (completedSteps < selfTrainSteps + 1L) {
      agent ! AgentSelfTrainRequest(
        TensorParallelDataset(
          "", Map(lang -> this.dataset.vocabulary(lang)), Map(lang -> Seq(nextDataset._2.next()._1))),
        StopCriteria.steps(trainStepsPerRequest))
    } else {
      val tensorsTuple = nextDataset._2.next()
      val tgtLang = nextDataset._1
      val vocabularies =  Map(lang -> this.dataset.vocabulary(lang), tgtLang -> this.dataset.vocabulary(tgtLang))
      val tensors =  Map(lang -> Seq(tensorsTuple._1), tgtLang -> Seq(tensorsTuple._2))
      agent ! AgentTrainRequest(
        agents(nextDataset._1), TensorParallelDataset("", vocabularies, tensors),
        StopCriteria.steps(trainStepsPerRequest))
    }
  }
}

object RoundRobinTrainScheduler {
  def apply(
      dataset: ParallelDataset,
      agents: Map[Language, ActorRef],
      selfTrainSteps: Long = 0L,
      trainStepsPerRequest: Long = 10L
  )(implicit
      sender: ActorRef = Actor.noSender
  ): RoundRobinTrainScheduler = {
    new RoundRobinTrainScheduler(dataset, agents, selfTrainSteps, trainStepsPerRequest)(sender)
  }
}
