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
import org.platanios.symphony.mt.data.{joinBilingualDatasets, ParallelDataset, TFBilingualDataset, TFMonolingualDataset}
import org.platanios.symphony.mt.models.Model
import org.platanios.symphony.mt.translators.actors.Messages._
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.StopCriteria

import akka.actor._

/**
  * @author Emmanouil Antonios Platanios
  */
class Agent protected (
    val language1: (Language, Vocabulary),
    val language2: (Language, Vocabulary),
    protected val model: (Language, Vocabulary, Language, Vocabulary) => Model,
    protected val requestManagerType: RequestManager.Type = RequestManager.Hash
) extends Actor with ActorLogging {
  protected val lang1ToLang2Model: Model = model(language1._1, language1._2, language2._1, language2._2)
  protected val lang2ToLang1Model: Model = model(language2._1, language2._2, language1._1, language1._2)

  /** Used for messages that map to stored request information. */
  protected var uniqueIdCounter: Long = 0L

  /** Used for storing requests associated with unique IDs. */
  protected val requestManager: RequestManager[Agent.RequestInformation] = {
    requestManagerType.newManager[Agent.RequestInformation]()
  }

  override def preStart(): Unit = {
    log.info(s"Translation agent for '${language1._1.abbreviation}-${language2._1.abbreviation}' started.")
  }

  override def postStop(): Unit = {
    log.info(s"Translation agent for '${language1._1.abbreviation}-${language2._1.abbreviation}' stopped.")
  }

  override def receive: Receive = {
    case Type =>
      sender() ! AgentActor(language1._1, language2._1)
    case AgentSelfTrainRequest(dataset, stopCriteria) =>
      processAgentSelfTrainRequest(dataset, stopCriteria)
    case AgentTrainRequest(tgtAgent, dataset, stopCriteria) =>
      processAgentTrainRequest(tgtAgent, dataset, stopCriteria)
    case AgentTranslateRequest(id, srcLanguage, tgtLanguage, dataset) =>
      processTranslateRequest(id, srcLanguage, tgtLanguage, dataset)
    case AgentTranslateResponse(id, language, sentences) =>
      processTranslateResponse(id, language, sentences)
  }

  @throws[IllegalArgumentException]
  protected def processAgentSelfTrainRequest(dataset: ParallelDataset, stopCriteria: StopCriteria): Unit = {
    require(language1._2.size == language2._2.size, "For self-training, all agent vocabularies must have same size.")

    // Train the translation models in both directions.
    lang1ToLang2Model.train(() => dataset.toTFBilingual(language1._1, language2._1, repeat = true), stopCriteria)
    lang2ToLang1Model.train(() => dataset.toTFBilingual(language2._1, language1._1, repeat = true), stopCriteria)

    // Send a message to the requester notifying that this agent is done processing this train request.
    sender() ! AgentSelfTrainResponse()
  }

  protected def processAgentTrainRequest(
      tgtAgent: ActorRef,
      dataset: ParallelDataset,
      stopCriteria: StopCriteria
  ): Unit = {
    requestManager.set(uniqueIdCounter, Agent.RequestInformation(sender(), dataset, Some(stopCriteria)))
    tgtAgent ! AgentTranslateRequest(uniqueIdCounter, language2._1, interlingua, dataset)
    uniqueIdCounter += 1
  }

  protected def processTranslateRequest(
      id: Long,
      srcLanguage: Language,
      tgtLanguage: Language,
      dataset: ParallelDataset
  ): Unit = {
    val translatedSentences = {
      if (srcLanguage == language1._1 && tgtLanguage == language2._1)
        lang1ToLang2Model.infer(() => dataset.toTFMonolingual(language1._1)).map(_._2)
      else if (srcLanguage == language2._1 && tgtLanguage == language1._1)
        lang2ToLang1Model.infer(() => dataset.toTFMonolingual(language2._1)).map(_._2)
      else throw new IllegalArgumentException(
        s"Agent '${self.path.name}' cannot translate from $srcLanguage to $tgtLanguage.")
    }
    sender() ! AgentTranslateResponse(id, tgtLanguage, translatedSentences)
  }

  protected def processTranslateResponse(
      id: Long,
      language: Language,
      sentences: Iterator[(Tensor, Tensor)]
  ): Unit = {
    requestManager.get(id) match {
      case Some(Agent.RequestInformation(requester, dataset, Some(trainStopCriteria))) =>
        // Train model for the human language to interlingua translation direction.
        lang1ToLang2Model.train(() =>
          dataset.toTFMonolingual(language1._1)
              .zip(joinBilingualDatasets(sentences.map(tf.data.TensorDataset(_): TFMonolingualDataset).toSeq).repeat())
              .asInstanceOf[TFBilingualDataset], trainStopCriteria)

        // Train model for the interlingua to human language translation direction.
        lang2ToLang1Model.train(() =>
          joinBilingualDatasets(sentences.map(tf.data.TensorDataset(_): TFMonolingualDataset).toSeq).repeat()
              .zip(dataset.toTFMonolingual(language1._1))
              .asInstanceOf[TFBilingualDataset], trainStopCriteria)

        // Send a message to the requester notifying that this agent is done processing this train request.
        requester ! AgentTrainResponse()
      case _ => log.warning(
        s"Ignoring translate response with ID '$id' because no relevant stored information was found.")
    }
  }
}

object Agent {
  def props(
      language1: (Language, Vocabulary),
      language2: (Language, Vocabulary),
      model: (Language, Vocabulary, Language, Vocabulary) => Model,
      requestManagerType: RequestManager.Type = RequestManager.Hash
  ): Props = Props(new Agent(language1, language2, model, requestManagerType))

  case class RequestInformation(
      requester: ActorRef,
      dataset: ParallelDataset,
      trainStopCriteria: Option[StopCriteria])
}
