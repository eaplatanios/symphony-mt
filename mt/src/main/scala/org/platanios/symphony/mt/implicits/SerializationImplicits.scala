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

package org.platanios.symphony.mt.implicits

import org.platanios.symphony.mt.Language

import better.files._
import io.circe._
import io.circe.generic.AutoDerivation

import java.nio.file.{Path, Paths}

/**
  * @author Emmanouil Antonios Platanios
  */
object SerializationImplicits extends AutoDerivation {
  implicit val encodePath: Encoder[Path] = new Encoder[Path] {
    final def apply(path: Path): Json = Json.fromString(path.toAbsolutePath.toString)
  }

  implicit val decodePath: Decoder[Path] = new Decoder[Path] {
    final def apply(cursor: HCursor): Decoder.Result[Path] =
      for (path <- cursor.as[String]) yield {
        Paths.get(path)
      }
  }

  implicit val encodeFile: Encoder[File] = new Encoder[File] {
    final def apply(file: File): Json = Json.fromString(file.path.toAbsolutePath.toString)
  }

  implicit val decodeFile: Decoder[File] = new Decoder[File] {
    final def apply(cursor: HCursor): Decoder.Result[File] =
      for (file <- cursor.as[String]) yield {
        File(file)
      }
  }

  implicit val encodeLanguage: Encoder[Language] = new Encoder[Language] {
    final def apply(language: Language): Json = Json.obj(
      ("name", Json.fromString(language.name)),
      ("abbreviation", Json.fromString(language.abbreviation))
    )
  }

  implicit val decodeLanguage: Decoder[Language] = new Decoder[Language] {
    final def apply(cursor: HCursor): Decoder.Result[Language] =
      for {
        name <- cursor.downField("name").as[String]
        abbreviation <- cursor.downField("abbreviation").as[String]
      } yield {
        Language(name, abbreviation)
      }
  }
}
