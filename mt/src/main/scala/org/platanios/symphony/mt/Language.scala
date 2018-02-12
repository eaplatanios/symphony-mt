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
  val bulgarian : Language = Language("Bulgarian", "bg")
  val czech     : Language = Language("Czech", "cs")
  val danish    : Language = Language("Danish", "da")
  val dutch     : Language = Language("Dutch", "nl")
  val english   : Language = Language("English", "en")
  val estonian  : Language = Language("Estonian", "et")
  val finnish   : Language = Language("Finnish", "fi")
  val french    : Language = Language("French", "fr")
  val german    : Language = Language("German", "de")
  val greek     : Language = Language("Greek", "el")
  val hindi     : Language = Language("Hindi", "hi")
  val hungarian : Language = Language("Hungarian", "hu")
  val italian   : Language = Language("Italian", "it")
  val lithuanian: Language = Language("Lithuanian", "lt")
  val latvian   : Language = Language("Latvian", "lv")
  val polish    : Language = Language("Polish", "pl")
  val portuguese: Language = Language("Portuguese", "pt")
  val romanian  : Language = Language("Romanian", "ro")
  val russian   : Language = Language("Russian", "ru")
  val slovak    : Language = Language("Slovak", "sk")
  val slovenian : Language = Language("Slovenian", "sl")
  val spanish   : Language = Language("Spanish", "es")
  val swedish   : Language = Language("Swedish", "sv")
  val turkish   : Language = Language("Turkish", "tr")
  val vietnamese: Language = Language("Vietnamese", "vi")
}
