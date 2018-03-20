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
case class Language(name: String, abbreviation: String) {
  override def toString: String = name
}

object Language {
  val Arabic          : Language = Language("Arabic", "ar")
  val Bulgarian       : Language = Language("Bulgarian", "bg")
  val Catalan         : Language = Language("Catalan", "ca")
  val Chinese         : Language = Language("Chinese", "zh")
  val Czech           : Language = Language("Czech", "cs")
  val Danish          : Language = Language("Danish", "da")
  val Dutch           : Language = Language("Dutch", "nl")
  val English         : Language = Language("English", "en")
  val Estonian        : Language = Language("Estonian", "et")
  val Finnish         : Language = Language("Finnish", "fi")
  val French          : Language = Language("French", "fr")
  val German          : Language = Language("German", "de")
  val Greek           : Language = Language("Greek", "el")
  val Hebrew          : Language = Language("Hebrew", "he")
  val Hindi           : Language = Language("Hindi", "hi")
  val Hungarian       : Language = Language("Hungarian", "hu")
  val Icelandic       : Language = Language("Icelandic", "is")
  val Italian         : Language = Language("Italian", "it")
  val Irish           : Language = Language("Irish", "ga")
  val Lithuanian      : Language = Language("Lithuanian", "lt")
  val Latvian         : Language = Language("Latvian", "lv")
  val Persian         : Language = Language("Persian", "fa")
  val Polish          : Language = Language("Polish", "pl")
  val Portuguese      : Language = Language("Portuguese", "pt")
  val PortugueseBrazil: Language = Language("PortugueseBrazil", "pt-br")
  val Romanian        : Language = Language("Romanian", "ro")
  val Russian         : Language = Language("Russian", "ru")
  val Slovak          : Language = Language("Slovak", "sk")
  val Slovenian       : Language = Language("Slovenian", "sl")
  val Spanish         : Language = Language("Spanish", "es")
  val Swedish         : Language = Language("Swedish", "sv")
  val Tamil           : Language = Language("Tamil", "ta")
  val Thai            : Language = Language("Thai", "th")
  val Turkish         : Language = Language("Turkish", "tr")
  val Vietnamese      : Language = Language("Vietnamese", "vi")
}
