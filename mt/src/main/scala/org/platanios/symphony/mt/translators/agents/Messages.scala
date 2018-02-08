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

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.data.LoadedDataset
import org.platanios.tensorflow.api.Tensor

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
    * @param  dataset Training dataset containing parallel corpora for (potentially) multiple language pairs.
    */
  case class SystemTrainRequest(dataset: LoadedDataset)

  // TODO: Use IDs for the system translate requests and responses.

  /** Message handled by a translation system, requesting a translation from some language to another.
    *
    * @param  srcLanguage Source language for the translation.
    * @param  tgtLanguage Target language for the translation.
    * @param  sentences   Tuple containing a padded tensor with word IDs in the source language and a tensor with the
    *                     sentence lengths. These are the sentences that need to be translated.
    */
  case class SystemTranslateRequest(srcLanguage: Language, tgtLanguage: Language, sentences: (Tensor, Tensor))

  /** Message sent by a translation system after having received a `SystemTranslateRequest` message, containing the
    * requested translations.
    *
    * @param  srcLanguage  Source language for the translation.
    * @param  tgtLanguage  Target language for the translation.
    * @param  srcSentences Tuple containing a padded tensor with word IDs in the source language and a tensor with the
    *                      sentence lengths. These are the sentences that need to be translated.
    * @param  tgtSentences Tuple containing a padded tensor with word IDs in the target language and a tensor with the
    *                      sentence lengths. These are the translated sentences for the request.
    *
    */
  case class SystemTranslateResponse(
      srcLanguage: Language,
      tgtLanguage: Language,
      srcSentences: (Tensor, Tensor),
      tgtSentences: (Tensor, Tensor))

  /** Message sent by a translation system to a translation agent, that provides it with a training pair for some
    * target language.
    *
    * @param  id                Unique ID for this request.
    * @param  tgtAgent          Translation agent responsible for the target language.
    * @param  parallelSentences Tuple containing parallel sentence examples. The first tuple element is a tuple
    *                           containing a padded tensor with word IDs in the agent's language and a tensor with the
    *                           sentence lengths. The second tuple element is the corresponding tuple for the sentences
    *                           in the target language.
    */
  case class AgentTrainRequest(id: Long, tgtAgent: ActorRef, parallelSentences: ((Tensor, Tensor), (Tensor, Tensor)))

  /** Message sent by a translation agent, once it has finished processing a train request.
    *
    * @param  id Unique ID that corresponds to the ID of the request for which this response is generated.
    */
  case class AgentTrainResponse(id: Long)

  /** Message sent to translation agents requesting a batch of sentences to be translated to the interlingua.
    *
    * @param  id        Unique ID for this request.
    * @param  sentences Tuple containing a padded tensor with word IDs in the agent's language and a tensor with the
    *                   sentence lengths. After receiving this message, the agent will translate the provided sentences
    *                   to the interlingua and respond back to the sender with an `AgentTranslateToInterlinguaResponse`
    *                   message.
    */
  case class AgentTranslateToInterlinguaRequest(id: Long, sentences: (Tensor, Tensor))

  /** Message sent by translation agents after having received an `AgentTranslateToInterlinguaRequest` message,
    * containing the requested translations.
    *
    * @param  id        Unique ID that corresponds to the ID of the request for which this response is generated.
    * @param  sentences Tuple containing a padded tensor with word IDs in the interlingua and a tensor with the sentence
    *                   lengths.
    */
  case class AgentTranslateToInterlinguaResponse(id: Long, sentences: (Tensor, Tensor))

  /** Message sent to translation agents requesting a batch of sentences to be translated from the interlingua to the
    * agent's language.
    *
    * @param  id        Unique ID for this request.
    * @param  sentences Tuple containing a padded tensor with word IDs in the interlingua and a tensor with the sentence
    *                   lengths. After receiving this message, the agent will translate the provided sentences from the
    *                   interlingua to it's own language and respond back to the sender with an
    *                   `AgentTranslateFromInterlinguaResponse` message.
    */
  case class AgentTranslateFromInterlinguaRequest(id: Long, sentences: (Tensor, Tensor))

  /** Message sent by translation agents after having received an `AgentTranslateFromInterlinguaRequest` message,
    * containing the requested translations.
    *
    * @param  id        Unique ID that corresponds to the ID of the request for which this response is generated.
    * @param  sentences Tuple containing a padded tensor with word IDs in the agent's language and a tensor with the
    *                   sentence lengths.
    */
  case class AgentTranslateFromInterlinguaResponse(id: Long, sentences: (Tensor, Tensor))
}
