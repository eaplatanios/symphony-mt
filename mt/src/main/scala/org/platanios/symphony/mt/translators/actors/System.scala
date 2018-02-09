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
import org.platanios.symphony.mt.data.LoadedDataset
import org.platanios.symphony.mt.models.Model
import org.platanios.symphony.mt.translators.actors.Messages._
import org.platanios.symphony.mt.vocabulary._
import org.platanios.tensorflow.api.Tensor

import akka.actor._
import better.files._
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
class System(
    val config: SystemConfig,
    protected val model: ((Language, Vocabulary), (Language, Vocabulary), Environment) => Model,
    protected val requestManagerType: RequestManager.Type = RequestManager.Hash
) extends Actor with ActorLogging {
  /** Working directory for this translation system. */
  protected val systemWorkingDir: File = File(config.environment.workingDir) / "system"

  /** Working directory for all translation agents. */
  protected val agentsWorkingDir: File = File(config.environment.workingDir) / "agents"

  /** State file for this translation system. */
  protected val systemStateFile: File = systemWorkingDir / "state.yaml"

  /** State for this translation system. */
  protected val systemState: SystemState = SystemState.load(systemStateFile) match {
    case Left(failure) =>
      System.logger.info(s"Translation system state file '$systemStateFile' could not be loaded.", failure)
      System.logger.info("A new translation system state file will be created.")
      val interlinguaVocabFile = systemWorkingDir / s"vocab.${Interlingua.abbreviation}"
      if (interlinguaVocabFile.notExists) {
        System.logger.info(s"Generating vocabulary file for $language.")
        DummyVocabularyGenerator(config.interlinguaVocabSize).generate(Seq.empty[File], interlinguaVocabFile)
        System.logger.info(s"Generated vocabulary file for $language.")
      }
      SystemState(
        interlinguaVocab = Vocabulary(interlinguaVocabFile),
        agents = Seq.empty[AgentState])
    case Right(state) =>
      System.logger.info(s"The translation system state file '$systemStateFile' was loaded.")
      state
  }

  /** Map containing the translation agents managed by this translation system. */
  protected val agents: mutable.Map[Language, ActorRef] = mutable.HashMap[Language, ActorRef](
    systemState.agents.map(agentState =>
      agentState.language -> createAgent(agentState.language, agentState.vocab, cleanWorkingDir = false)): _*)

  /** Used for messages that map to stored request information. */
  protected var uniqueIdCounter: Long = 0L

  /** Used for storing requests associated with unique IDs. */
  protected val requestManager: RequestManager[System.RequestInformation] = {
    requestManagerType.newManager[System.RequestInformation]()
  }

  // TODO: Make this configurable.
  protected var trainScheduler: TrainScheduler = _

  override def preStart(): Unit = log.info("Translation system started.")
  override def postStop(): Unit = log.info("Translation system stopped.")

  override def receive: Receive = {
    case Type =>
      sender() ! SystemActor
    case SystemTrainRequest(dataset) =>
      processSystemTrainRequest(dataset)
    case SystemTranslateRequest(srcLang, tgtLang, sentences) =>
      processSystemTranslateRequest(sender(), srcLang, tgtLang, sentences)
    case AgentTranslateToInterlinguaResponse(id, sentences) =>
      processAgentTranslateToInterlinguaResponse(id, sentences)
    case AgentTranslateFromInterlinguaResponse(id, sentences) =>
      processAgentTranslateFromInterlinguaResponse(id, sentences)
  }

  protected def processSystemTrainRequest(dataset: LoadedDataset): Unit = {
    dataset.vocabularies.foreach {
      case (lang, vocab) => agents.getOrElseUpdate(lang, createAgent(lang, vocab, cleanWorkingDir = true))
    }
    // TODO: Make this configurable.
    trainScheduler = RoundRobinTrainScheduler(dataset, agents.toMap)
  }

  @throws[IllegalArgumentException]
  protected def processSystemTranslateRequest(
      sender: ActorRef,
      srcLang: Language,
      tgtLang: Language,
      sentences: (Tensor, Tensor)
  ): Unit = {
    requestManager.set(uniqueIdCounter, System.RequestInformation(sender, srcLang, tgtLang, sentences))
    if (!agents.contains(srcLang))
      throw new IllegalArgumentException(s"No training data have been provided for language '$srcLang'.")
    if (!agents.contains(tgtLang))
      throw new IllegalArgumentException(s"No training data have been provided for language '$tgtLang'.")
    agents(srcLang) ! AgentTranslateToInterlinguaRequest(uniqueIdCounter, sentences)
    uniqueIdCounter += 1
  }

  @throws[IllegalArgumentException]
  protected def processAgentTranslateToInterlinguaResponse(id: Long, sentences: (Tensor, Tensor)): Unit = {
    requestManager.get(id, remove = false) match {
      case Some(System.RequestInformation(_, _, tgtLang, _)) =>
        if (!agents.contains(tgtLang))
          throw new IllegalArgumentException(s"No training data have been provided for language '$tgtLang'.")
        agents(tgtLang) ! AgentTranslateFromInterlinguaRequest(id, sentences)
      case None => log.warning(
        s"Ignoring translate-to-interlingua response with ID '$id' " +
            s"because no relevant stored information was found.")
    }
  }

  protected def processAgentTranslateFromInterlinguaResponse(id: Long, sentences: (Tensor, Tensor)): Unit = {
    requestManager.get(id, remove = false) match {
      case Some(System.RequestInformation(requester, srcLang, tgtLang, srcSentences)) =>
        requester ! SystemTranslateResponse(srcLang, tgtLang, srcSentences, sentences)
      case None => log.warning(
        s"Ignoring translate-from-interlingua response with ID '$id' " +
            s"because no relevant stored information was found.")
    }
  }

  protected def createAgent(lang: Language, vocab: Vocabulary, cleanWorkingDir: Boolean = false): ActorRef = {
    val workingDir = agentsWorkingDir / lang.abbreviation
    if (cleanWorkingDir && workingDir.exists)
      workingDir.delete()
    workingDir.createIfNotExists(asDirectory = true, createParents = true)
    agents.getOrElseUpdate(lang, context.actorOf(
      Agent.props(
        lang, vocab, systemState.interlinguaVocab,
        model(_, _, config.environment.copy(workingDir = workingDir.path)),
        requestManagerType), s"translation-agent-${lang.abbreviation}"))
  }
}

object System {
  private[actors] val logger = Logger(LoggerFactory.getLogger("Translation System"))

  def props(
      config: SystemConfig,
      model: ((Language, Vocabulary), (Language, Vocabulary), Environment) => Model,
      requestManagerType: RequestManager.Type = RequestManager.Hash
  ): Props = {
    Props(new System(config, model, requestManagerType))
  }

  case class RequestInformation(
      requester: ActorRef,
      srcLang: Language,
      tgtLanguage: Language,
      sentences: (Tensor, Tensor))
}
