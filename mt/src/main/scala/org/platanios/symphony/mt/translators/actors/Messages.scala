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
import org.platanios.tensorflow.api.Tensor
import org.platanios.tensorflow.api.learn.StopCriteria

import akka.actor.ActorRef

/** Contains all the messages used by the translation system and agent actors.
  *
  * @author Emmanouil Antonios Platanios
  */
object Messages {
  /** Message sent to any actor, requesting for a response containing the agent's type. */
  case object Type

  /** Message handled by a translation system, that provides it with a training dataset.
    *
    * @param  dataset      Training dataset containing parallel corpora for (potentially) multiple language pairs.
    * @param  stopCriteria
    */
  case class SystemTrainRequest(dataset: ParallelDataset, stopCriteria: StopCriteria)

  // TODO: Use IDs for the system translate requests and responses.

  /** Message handled by a translation system, requesting a translation from some language to another.
    *
    * @param  srcLanguage Source language for the translation.
    * @param  tgtLanguage Target language for the translation.
    * @param  dataset     Dataset to translate.
    */
  case class SystemTranslateRequest(srcLanguage: Language, tgtLanguage: Language, dataset: ParallelDataset)

  /** Message sent by a translation system after having received a `SystemTranslateRequest` message, containing the
    * requested translations.
    *
    * @param  srcLanguage  Source language for the translation.
    * @param  tgtLanguage  Target language for the translation.
    * @param  dataset      Original translation request dataset.
    * @param  tgtSentences Tuple containing a padded tensor with word IDs in the target language and a tensor with the
    *                      sentence lengths. These are the translated sentences for the request.
    *
    */
  case class SystemTranslateResponse(
      srcLanguage: Language,
      tgtLanguage: Language,
      dataset: ParallelDataset, // TODO: Switch to not returning `srcSentences` when we start using a request ID.
      tgtSentences: Iterator[(Tensor, Tensor)])

  /** Message sent by a translation system to a translation agent, that provides it with sentences in its own language,
    * so that it learns to translate between its own language and the interlingua.
    *
    * @param  dataset      Dataset to use for self-training.
    * @param  stopCriteria Stop criteria to use for this train request.
    */
  case class AgentSelfTrainRequest(dataset: ParallelDataset, stopCriteria: StopCriteria)

  /** Message sent by a translation agent, once it has finished processing a self-train request. */
  case class AgentSelfTrainResponse()

  /** Message sent by a translation system to a translation agent, that provides it with a training pair for some
    * target language.
    *
    * @param  tgtAgent     Translation agent responsible for the target language.
    * @param  dataset      Dataset to use for training.
    * @param  stopCriteria Stop criteria to use for this train request.
    */
  case class AgentTrainRequest(tgtAgent: ActorRef, dataset: ParallelDataset, stopCriteria: StopCriteria)

  /** Message sent by a translation agent, once it has finished processing a train request. */
  case class AgentTrainResponse()

  /** Message sent to translation agents requesting a batch of sentences to be translated to the interlingua.
    *
    * @param  id          Unique ID for this request.
    * @param  srcLanguage
    * @param  tgtLanguage
    * @param  dataset     Dataset to translate.
    */
  case class AgentTranslateRequest(id: Long, srcLanguage: Language, tgtLanguage: Language, dataset: ParallelDataset)

  /** Message sent by translation agents after having received an `AgentTranslateToInterlinguaRequest` message,
    * containing the requested translations.
    *
    * @param  id        Unique ID that corresponds to the ID of the request for which this response is generated.
    * @param  sentences Iterator over translated sentences. Each element is a tuple that contains a padded tensor with
    *                   word IDs in the interlingua and a tensor with the sentence lengths.
    */
  case class AgentTranslateResponse(id: Long, language: Language, sentences: Iterator[(Tensor, Tensor)])
}
