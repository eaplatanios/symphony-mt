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

package org.platanios.symphony.mt

import better.files._

import java.io.{BufferedReader, BufferedWriter, InputStreamReader, OutputStreamWriter}
import java.nio.charset._
import java.nio.file.StandardOpenOption

package object data {
  private[this] val encoder: CharsetEncoder = StandardCharsets.UTF_8.newEncoder()
  private[this] val decoder: CharsetDecoder = StandardCharsets.UTF_8.newDecoder()

  decoder.onMalformedInput(CodingErrorAction.IGNORE)

  private[mt] def newReader(file: File): BufferedReader = {
    new BufferedReader(
      new InputStreamReader(
        file.newInputStream(),
        decoder))
  }

  private[mt] def newWriter(file: File): BufferedWriter = {
    new BufferedWriter(
      new OutputStreamWriter(
        file.newOutputStream(Seq(
          StandardOpenOption.CREATE,
          StandardOpenOption.WRITE,
          StandardOpenOption.TRUNCATE_EXISTING)),
        encoder))
  }
}
