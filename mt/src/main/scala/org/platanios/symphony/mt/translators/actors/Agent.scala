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
import org.platanios.symphony.mt.data.{MTInferDataset, MTTrainDataset}
import org.platanios.symphony.mt.models.Model
import org.platanios.symphony.mt.translators.actors.Messages._
import org.platanios.symphony.mt.vocabulary.Vocabulary
import org.platanios.tensorflow.api._

import akka.actor._

/**
  * @author Emmanouil Antonios Platanios
  */
class Agent(
    val language: Language,
    protected val languageVocab: Vocabulary,
    protected val interlinguaVocab: Vocabulary,
    protected val model: ((Language, Vocabulary), (Language, Vocabulary)) => Model,
    protected val requestManagerType: RequestManager.Type = RequestManager.Hash
) extends Actor with ActorLogging {
  protected val langToInterlinguaModel: Model = model((language, languageVocab), (Interlingua, interlinguaVocab))
  protected val interlinguaToLangModel: Model = model((Interlingua, interlinguaVocab), (language, languageVocab))

  /** Used for messages that map to stored request information. */
  protected var uniqueIdCounter: Long = 0L

  /** Used for storing requests associated with unique IDs. */
  protected val requestManager: RequestManager[Agent.RequestInformation] = {
    requestManagerType.newManager[Agent.RequestInformation]()
  }

  override def preStart(): Unit = log.info(s"Translation agent for '$language' started.")
  override def postStop(): Unit = log.info(s"Translation agent for '$language' stopped.")

  override def receive: Receive = {
    case Type =>
      sender() ! AgentActor(language)
    case AgentTrainRequest(tgtAgent, parallelSentences) =>
      processAgentTrainRequest(tgtAgent, parallelSentences)
    case AgentTranslateToInterlinguaRequest(id, sentences) =>
      processTranslateToInterlinguaRequest(id, sentences)
    case AgentTranslateToInterlinguaResponse(id, interlinguaSentences) =>
      processTranslateToInterlinguaResponse(id, interlinguaSentences)
    case AgentTranslateFromInterlinguaRequest(id, interlinguaSentences) =>
      processTranslateFromInterlinguaRequest(id, interlinguaSentences)
    case AgentTranslateFromInterlinguaResponse(id, sentences) =>
      processTranslateFromInterlinguaResponse(id, sentences)
  }

  protected def processAgentTrainRequest(
      tgtAgent: ActorRef,
      parallelSentences: ((Tensor, Tensor), (Tensor, Tensor))
  ): Unit = {
    requestManager.set(
      uniqueIdCounter, Agent.RequestInformation(sender(), parallelSentences._1, isTrain = true))
    tgtAgent ! AgentTranslateToInterlinguaRequest(uniqueIdCounter, parallelSentences._2)
    uniqueIdCounter += 1
  }

  protected def processTranslateToInterlinguaRequest(id: Long, sentences: (Tensor, Tensor)): Unit = {
    val translatedSentences = langToInterlinguaModel.infer(
      () => tf.data.TensorDataset(sentences).repeat().asInstanceOf[MTInferDataset]).next()._2
    sender() ! AgentTranslateToInterlinguaResponse(id, translatedSentences)
  }

  protected def processTranslateToInterlinguaResponse(id: Long, interlinguaSentences: (Tensor, Tensor)): Unit = {
    requestManager.get(id) match {
      case Some(Agent.RequestInformation(requester, srcSentences, isTrain)) =>
        if (isTrain) {
          // Train model for the human language to interlingua translation direction.
          langToInterlinguaModel.train(() => tf.data.TensorDataset(
            (srcSentences, interlinguaSentences)).repeat().asInstanceOf[MTTrainDataset], ???)

          // Train model for the interlingua to human language translation direction.
          interlinguaToLangModel.train(() => tf.data.TensorDataset(
            (interlinguaSentences, srcSentences)).repeat().asInstanceOf[MTTrainDataset], ???)

          // Send a message to the requester notifying that this agent is done processing this train request.
          requester ! AgentTrainResponse()
        } else {
          ??? // TODO: Impossible case as `AgentTranslateToInterlinguaRequest` is only sent from within the train request.
        }
      case None => log.warning(
        s"Ignoring translate-to-interlingua response with ID '$id' " +
            s"because no relevant stored information was found.")
    }
  }

  protected def processTranslateFromInterlinguaRequest(id: Long, interlinguaSentences: (Tensor, Tensor)): Unit = {
    val translatedSentences = interlinguaToLangModel.infer(
      () => tf.data.TensorDataset(
        interlinguaSentences).repeat().asInstanceOf[MTInferDataset]).next()._2
    sender() ! AgentTranslateFromInterlinguaResponse(id, translatedSentences)
  }

  protected def processTranslateFromInterlinguaResponse(id: Long, sentences: (Tensor, Tensor)): Unit = {
    ??? // TODO: Impossible case as `AgentTranslateFromInterlinguaRequest` is only sent from translation systems.
  }
}

object Agent {
  def props(
      language: Language,
      languageVocab: Vocabulary,
      interlinguaVocab: Vocabulary,
      model: ((Language, Vocabulary), (Language, Vocabulary)) => Model,
      requestManagerType: RequestManager.Type = RequestManager.Hash
  ): Props = Props(new Agent(language, languageVocab, interlinguaVocab, model, requestManagerType))

  case class RequestInformation(requester: ActorRef, sentences: (Tensor, Tensor), isTrain: Boolean)
}
