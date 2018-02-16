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

import org.platanios.symphony.mt.{Environment, Language}
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.models.Model
import org.platanios.symphony.mt.translators.actors.Messages._
import org.platanios.symphony.mt.vocabulary._
import org.platanios.tensorflow.api.Tensor
import org.platanios.tensorflow.api.learn.StopCriteria

import akka.actor._
import better.files._
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
class BilingualSystem protected (
    val config: SystemConfig,
    protected val model: (Language, Vocabulary, Language, Vocabulary, Environment) => Model,
    protected val requestManagerType: RequestManager.Type = RequestManager.Hash
) extends Actor with ActorLogging {
  /** Working directory for this translation system. */
  protected val systemWorkingDir: File = File(config.env.workingDir) / "system"

  /** Working directory for all translation agents. */
  protected val agentsWorkingDir: File = File(config.env.workingDir) / "agents"

  /** State file for this translation system. */
  protected val systemStateFile: File = systemWorkingDir / "state.yaml"

  /** State for this translation system. */
  protected var systemState: SystemState = SystemState.load(systemStateFile) match {
    case Left(failure) =>
      System.logger.info(s"Translation system state file '$systemStateFile' could not be loaded.", failure.getMessage)
      System.logger.info("A new bilingual translation system state file will be created.")
      SystemState(
        interlinguaVocab = None,
        agents = Seq.empty[AgentState])
    case Right(state) =>
      System.logger.info(s"The translation system state file '$systemStateFile' was loaded.")
      state
  }

  /** Map containing the translation agents managed by this translation system. */
  protected val agents: mutable.Map[(Language, Language), ActorRef] = {
    mutable.HashMap.empty[(Language, Language), ActorRef]
  }

  // Initialize the agents map from the current system state (in case it's been loaded from a file).
  systemState.agents.foreach(agentState =>
    createAgent(agentState.language1, agentState.language2, cleanWorkingDir = false))

  /** Used for messages that map to stored request information. */
  protected var uniqueIdCounter: Long = 0L

  /** Used for storing requests associated with unique IDs. */
  protected val requestManager: RequestManager[System.RequestInformation] = {
    requestManagerType.newManager[System.RequestInformation]()
  }

  override def preStart(): Unit = log.info("Translation system started.")
  override def postStop(): Unit = log.info("Translation system stopped.")

  override def receive: Receive = {
    case Type =>
      sender() ! SystemActor
    case SystemTrainRequest(dataset, stopCriteria) =>
      processSystemTrainRequest(dataset, stopCriteria)
    case AgentSelfTrainResponse() => ???
    case AgentTrainResponse() => ???
    case SystemTranslateRequest(srcLanguage, tgtLanguage, dataset) =>
      processSystemTranslateRequest(srcLanguage, tgtLanguage, dataset)
    case AgentTranslateResponse(id, language, sentences) =>
      processAgentTranslateResponse(id, language, sentences)
  }

  protected def processSystemTrainRequest(
      dataset: ParallelDataset,
      stopCriteria: StopCriteria
  ): Unit = {
    dataset.languagePairs(includeReversed = false).foreach {
      case (language1, language2) =>
        val agent = agents.getOrElseUpdate((language1, language2), {
          val vocabulary1 = dataset.vocabulary(language1)
          val vocabulary2 = dataset.vocabulary(language2)
          val agent = createAgent(language1 -> vocabulary1, language2 -> vocabulary2, cleanWorkingDir = true)
          systemState = systemState.copy(agents = systemState.agents :+
              AgentState(language1 -> vocabulary1, language2 -> vocabulary2))
          SystemState.save(systemState, systemStateFile)
          agent
        })
        agent ! AgentTrainRequest(agent, dataset, stopCriteria)
    }
  }

  @throws[IllegalArgumentException]
  protected def processSystemTranslateRequest(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataset: ParallelDataset
  ): Unit = {
    requestManager.set(uniqueIdCounter, System.RequestInformation(sender(), srcLanguage, tgtLanguage, dataset))
    if (!agents.contains((srcLanguage, tgtLanguage)))
      throw new IllegalArgumentException(s"No training data exist for language pair $srcLanguage - $tgtLanguage.")
    agents((srcLanguage, tgtLanguage)) ! AgentTranslateRequest(uniqueIdCounter, srcLanguage, tgtLanguage, dataset)
    uniqueIdCounter += 1
  }

  @throws[IllegalArgumentException]
  protected def processAgentTranslateResponse(
      id: Long,
      language: Language,
      sentences: Seq[(Tensor, Tensor)]
  ): Unit = {
    requestManager.get(id, remove = false) match {
      case Some(System.RequestInformation(requester, srcLanguage, tgtLanguage, dataset)) if language == tgtLanguage =>
        requester ! SystemTranslateResponse(srcLanguage, tgtLanguage, dataset, sentences)
      case Some(System.RequestInformation(_, _, tgtLanguage, dataset)) =>
        if (!agents.contains((language, tgtLanguage)))
          throw new IllegalArgumentException(s"No training data exist for language pair $language - $tgtLanguage.")
        // TODO: !!! Make this more efficient. Creating new datasets can have an overhead.
        agents((language, tgtLanguage)) ! AgentTranslateRequest(id, language, tgtLanguage, TensorParallelDataset(
          "", Map(language -> dataset.vocabulary(language)), Map(language -> sentences.toSeq)))
      case None => log.warning(
        s"Ignoring agent translate response with ID '$id' because no relevant stored information was found.")
    }
  }

  protected def createAgent(
      language1: (Language, Vocabulary),
      language2: (Language, Vocabulary),
      cleanWorkingDir: Boolean = false
  ): ActorRef = {
    val languagePair = s"${language1._1.abbreviation}-${language2._1.abbreviation}"
    val workingDir = agentsWorkingDir / languagePair
    if (cleanWorkingDir && workingDir.exists)
      workingDir.delete()
    workingDir.createIfNotExists(asDirectory = true, createParents = true)
    agents.getOrElseUpdate((language1._1, language2._1), {
      val agent = context.actorOf(
        Agent.props(
          language1, language2, model(_, _, _, _, config.env.copy(workingDir = workingDir.path)),
          requestManagerType), s"translation-agent-$languagePair")
      agents.values.foreach(_ ! AgentForLanguagePair(language1._1, language2._1, agent))
      agents.values.foreach(_ ! AgentForLanguagePair(language2._1, language1._1, agent))
      agents.foreach {
        case ((l1, l2), a) =>
          agent ! AgentForLanguagePair(l1, l2, a)
          agent ! AgentForLanguagePair(l2, l1, a)
      }
      agent
    })
  }
}

object BilingualSystem {
  private[actors] val logger = Logger(LoggerFactory.getLogger("Bilingual Translation System"))

  def props(
      config: SystemConfig,
      model: (Language, Vocabulary, Language, Vocabulary, Environment) => Model,
      requestManagerType: RequestManager.Type = RequestManager.Hash
  ): Props = {
    Props(new BilingualSystem(config, model, requestManagerType))
  }

  case class RequestInformation(
      requester: ActorRef,
      srcLanguage: Language,
      tgtLanguage: Language,
      dataset: ParallelDataset)
}
