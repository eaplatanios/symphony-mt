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

import org.platanios.symphony.mt.implicits.SerializationImplicits._
import org.platanios.symphony.mt.vocabulary.Vocabulary

import better.files.File
import io.circe.syntax._
import io.circe.yaml.{parser => YAMLParser}
import io.circe.yaml.syntax._

import java.io.FileNotFoundException

/**
  * @author Emmanouil Antonios Platanios
  */
case class SystemState(interlinguaVocab: Option[Vocabulary], agents: Seq[AgentState])

object SystemState {
  def save(state: SystemState, file: File): File = {
    file.createIfNotExists(createParents = true)
    file.overwrite(state.asJson.asYaml.spaces2)
  }

  def load(file: File): Either[Throwable, SystemState] = {
    if (file.notExists)
      Left(new FileNotFoundException(s"'$file' was not found."))
    else
      YAMLParser.parse(file.contentAsString).flatMap(_.as[SystemState])
  }
}
