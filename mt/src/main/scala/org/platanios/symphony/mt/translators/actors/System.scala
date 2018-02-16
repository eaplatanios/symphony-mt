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
import org.platanios.symphony.mt.data.{ParallelDataset, TensorParallelDataset}
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
class System protected (
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
      System.logger.info("A new translation system state file will be created.")
      val interlinguaVocabFile = systemWorkingDir / s"vocab.${interlingua.abbreviation}"
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
  protected val agents: mutable.Map[Language, ActorRef] = mutable.HashMap.empty[Language, ActorRef]

  // Initialize the agents map from the current system state (in case it's been loaded from a file).
  systemState.agents.foreach(agentState => createAgent(agentState.language1, cleanWorkingDir = false))

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
    case SystemTrainRequest(dataset, stopCriteria) =>
      processSystemTrainRequest(dataset, stopCriteria)
    case AgentSelfTrainResponse() =>
      trainScheduler.onTrainResponse(sender())
    case AgentTrainResponse() =>
      trainScheduler.onTrainResponse(sender())
    case SystemTranslateRequest(srcLanguage, tgtLanguage, dataset) =>
      processSystemTranslateRequest(srcLanguage, tgtLanguage, dataset)
    case AgentTranslateResponse(id, language, sentences) =>
      processAgentTranslateResponse(id, language, sentences)
  }

  protected def processSystemTrainRequest(
      dataset: ParallelDataset,
      stopCriteria: StopCriteria // TODO: !!! Use the stop criteria.
  ): Unit = {
    dataset.vocabulary.foreach {
      case (lang, vocab) => agents.getOrElseUpdate(lang, {
        val agent = createAgent(lang -> vocab, cleanWorkingDir = true)
        systemState = systemState.copy(agents = systemState.agents :+
            AgentState(lang -> vocab, interlingua -> systemState.interlinguaVocab))
        SystemState.save(systemState, systemStateFile)
        agent
      })
    }
    // TODO: Make this configurable.
    trainScheduler = RoundRobinTrainScheduler(
      dataset, agents.toMap,
      selfTrainSteps = config.selfTrainSteps,
      trainStepsPerRequest = config.trainStepsPerRequest)
    trainScheduler.initialize()
  }

  @throws[IllegalArgumentException]
  protected def processSystemTranslateRequest(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataset: ParallelDataset
  ): Unit = {
    requestManager.set(uniqueIdCounter, System.RequestInformation(sender(), srcLanguage, tgtLanguage, dataset))
    if (!agents.contains(srcLanguage))
      throw new IllegalArgumentException(s"No training data have been provided for language '$srcLanguage'.")
    if (!agents.contains(tgtLanguage))
      throw new IllegalArgumentException(s"No training data have been provided for language '$tgtLanguage'.")
    agents(srcLanguage) ! AgentTranslateRequest(uniqueIdCounter, srcLanguage, interlingua, dataset)
    uniqueIdCounter += 1
  }

  @throws[IllegalArgumentException]
  protected def processAgentTranslateResponse(
      id: Long,
      language: Language,
      sentences: Iterator[(Tensor, Tensor)]
  ): Unit = {
    requestManager.get(id, remove = false) match {
      case Some(System.RequestInformation(requester, srcLanguage, tgtLanguage, dataset)) if language == tgtLanguage =>
        requester ! SystemTranslateResponse(srcLanguage, tgtLanguage, dataset, sentences)
      case Some(System.RequestInformation(_, _, tgtLanguage, _)) =>
        if (!agents.contains(tgtLanguage))
          throw new IllegalArgumentException(s"No training data have been provided for language '$tgtLanguage'.")
        // TODO: !!! Make this more efficient. Creating new datasets can have an overhead.
        agents(tgtLanguage) ! AgentTranslateRequest(id, interlingua, tgtLanguage, TensorParallelDataset(
          "", Map(interlingua -> systemState.interlinguaVocab), Map(interlingua -> sentences.toSeq)))
      case None => log.warning(
        s"Ignoring agent translate response with ID '$id' because no relevant stored information was found.")
    }
  }

  protected def createAgent(
      language: (Language, Vocabulary),
      cleanWorkingDir: Boolean = false
  ): ActorRef = {
    val languagePair = s"${language._1.abbreviation}-${interlingua.abbreviation}"
    val workingDir = agentsWorkingDir / languagePair
    if (cleanWorkingDir && workingDir.exists)
      workingDir.delete()
    workingDir.createIfNotExists(asDirectory = true, createParents = true)
    agents.getOrElseUpdate(language._1, context.actorOf(
      Agent.props(
        language, interlingua -> systemState.interlinguaVocab,
        model(_, _, _, _, config.env.copy(workingDir = workingDir.path)),
        requestManagerType), s"translation-agent-$languagePair"))
  }
}

object System {
  private[actors] val logger = Logger(LoggerFactory.getLogger("Translation System"))

  def props(
      config: SystemConfig,
      model: (Language, Vocabulary, Language, Vocabulary, Environment) => Model,
      requestManagerType: RequestManager.Type = RequestManager.Hash
  ): Props = {
    Props(new System(config, model, requestManagerType))
  }

  case class RequestInformation(
      requester: ActorRef,
      srcLanguage: Language,
      tgtLanguage: Language,
      dataset: ParallelDataset)
}
