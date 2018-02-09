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

/**
  * @author Emmanouil Antonios Platanios
  */
class Language protected (val name: String, val abbreviation: String) {
  override def toString: String = name
}

object Language {
  def apply(name: String, abbreviation: String): Language = new Language(name, abbreviation)

  case object Bulgarian extends Language("Bulgarian", "bg")
  case object Czech extends Language("Czech", "cs")
  case object Danish extends Language("Danish", "da")
  case object Dutch extends Language("Dutch", "nl")
  case object English extends Language("English", "en")
  case object Estonian extends Language("Estonian", "et")
  case object Finnish extends Language("Finnish", "fi")
  case object French extends Language("French", "fr")
  case object German extends Language("German", "de")
  case object Greek extends Language("Greek", "el")
  case object Hindi extends Language("Hindi", "hi")
  case object Hungarian extends Language("Hungarian", "hu")
  case object Italian extends Language("Italian", "it")
  case object Lithuanian extends Language("Lithuanian", "lt")
  case object Latvian extends Language("Latvian", "lv")
  case object Polish extends Language("Polish", "pl")
  case object Portuguese extends Language("Portuguese", "pt")
  case object Romanian extends Language("Romanian", "ro")
  case object Russian extends Language("Russian", "ru")
  case object Slovak extends Language("Slovak", "sk")
  case object Slovenian extends Language("Slovenian", "sl")
  case object Spanish extends Language("Spanish", "es")
  case object Swedish extends Language("Swedish", "sv")
  case object Turkish extends Language("Turkish", "tr")
  case object Vietnamese extends Language("Vietnamese", "vi")
}
