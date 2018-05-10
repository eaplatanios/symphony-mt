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
  val Albanian        : Language = Language("Albanian", "sq")
  val Arabic          : Language = Language("Arabic", "ar")
  val Armenian        : Language = Language("Armenian", "hy")
  val Azerbaijani     : Language = Language("Azerbaijani", "az")
  val Basque          : Language = Language("Basque", "eu")
  val Belarusian      : Language = Language("Belarusian", "be")
  val Bengali         : Language = Language("Bengali", "bn")
  val Bosnian         : Language = Language("Bosnian", "bs")
  val Bulgarian       : Language = Language("Bulgarian", "bg")
  val Burmese         : Language = Language("Burmese", "my")
  val Catalan         : Language = Language("Catalan", "ca")
  val Chinese         : Language = Language("Chinese", "zh")
  val ChineseMainland : Language = Language("ChineseMainland", "zh-cn")
  val ChineseTaiwan   : Language = Language("ChineseTaiwan", "zh-tw")
  val Croatian        : Language = Language("Croatian", "hr")
  val Czech           : Language = Language("Czech", "cs")
  val Danish          : Language = Language("Danish", "da")
  val Dutch           : Language = Language("Dutch", "nl")
  val English         : Language = Language("English", "en")
  val Esperanto       : Language = Language("Esperanto", "eo")
  val Estonian        : Language = Language("Estonian", "et")
  val Finnish         : Language = Language("Finnish", "fi")
  val French          : Language = Language("French", "fr")
  val FrenchCanada    : Language = Language("FrenchCanada", "fr-ca")
  val Galician        : Language = Language("Galician", "gl")
  val German          : Language = Language("German", "de")
  val Georgian        : Language = Language("Georgian", "ka")
  val Greek           : Language = Language("Greek", "el")
  val Hebrew          : Language = Language("Hebrew", "he")
  val Hindi           : Language = Language("Hindi", "hi")
  val Hungarian       : Language = Language("Hungarian", "hu")
  val Icelandic       : Language = Language("Icelandic", "is")
  val Indonesian      : Language = Language("Indonesian", "id")
  val Italian         : Language = Language("Italian", "it")
  val Irish           : Language = Language("Irish", "ga")
  val Japanese        : Language = Language("Japanese", "ja")
  val Kazakh          : Language = Language("Kazakh", "kk")
  val Korean          : Language = Language("Korean", "ko")
  val Kurdish         : Language = Language("Kurdish", "ku")
  val Lithuanian      : Language = Language("Lithuanian", "lt")
  val Latvian         : Language = Language("Latvian", "lv")
  val Macedonian      : Language = Language("Macedonian", "mk")
  val Malay           : Language = Language("Malay", "ms")
  val Marathi         : Language = Language("Marathi", "mr")
  val Mongolian       : Language = Language("Mongolian", "mn")
  val Norwegian       : Language = Language("Norwegian", "nb")
  val Persian         : Language = Language("Persian", "fa")
  val Polish          : Language = Language("Polish", "pl")
  val Portuguese      : Language = Language("Portuguese", "pt")
  val PortugueseBrazil: Language = Language("PortugueseBrazil", "pt-br")
  val Romanian        : Language = Language("Romanian", "ro")
  val Russian         : Language = Language("Russian", "ru")
  val Serbian         : Language = Language("Serbian", "sr")
  val Slovak          : Language = Language("Slovak", "sk")
  val Slovenian       : Language = Language("Slovenian", "sl")
  val Spanish         : Language = Language("Spanish", "es")
  val Swedish         : Language = Language("Swedish", "sv")
  val Tamil           : Language = Language("Tamil", "ta")
  val Thai            : Language = Language("Thai", "th")
  val Turkish         : Language = Language("Turkish", "tr")
  val Ukranian        : Language = Language("Ukranian", "uk")
  val Urdu            : Language = Language("Urdu", "ur")
  val Vietnamese      : Language = Language("Vietnamese", "vi")

  @throws[IllegalArgumentException]
  def fromName(name: String): Language = name match {
    case "Albanian" => Albanian
    case "Arabic" => Arabic
    case "Armenian" => Armenian
    case "Azerbaijani" => Azerbaijani
    case "Basque" => Basque
    case "Belarusian" => Belarusian
    case "Bengali" => Bengali
    case "Bosnian" => Bosnian
    case "Bulgarian" => Bulgarian
    case "Burmese" => Burmese
    case "Catalan" => Catalan
    case "Chinese" => Chinese
    case "ChineseMainland" => ChineseMainland
    case "ChineseTaiwan" => ChineseTaiwan
    case "Croatian" => Croatian
    case "Czech" => Czech
    case "Danish" => Danish
    case "Dutch" => Dutch
    case "English" => English
    case "Esperanto" => Esperanto
    case "Estonian" => Estonian
    case "Finnish" => Finnish
    case "French" => French
    case "FrenchCanada" => FrenchCanada
    case "Galician" => Galician
    case "German" => German
    case "Georgian" => Georgian
    case "Greek" => Greek
    case "Hebrew" => Hebrew
    case "Hindi" => Hindi
    case "Hungarian" => Hungarian
    case "Icelandic" => Icelandic
    case "Indonesian" => Indonesian
    case "Italian" => Italian
    case "Irish" => Irish
    case "Japanese" => Japanese
    case "Kazakh" => Kazakh
    case "Korean" => Korean
    case "Kurdish" => Kurdish
    case "Lithuanian" => Lithuanian
    case "Latvian" => Latvian
    case "Macedonian" => Macedonian
    case "Malay" => Malay
    case "Marathi" => Marathi
    case "Mongolian" => Mongolian
    case "Norwegian" => Norwegian
    case "Persian" => Persian
    case "Polish" => Polish
    case "Portuguese" => Portuguese
    case "PortugueseBrazil" => PortugueseBrazil
    case "Romanian" => Romanian
    case "Russian" => Russian
    case "Serbian" => Serbian
    case "Slovak" => Slovak
    case "Slovenian" => Slovenian
    case "Spanish" => Spanish
    case "Swedish" => Swedish
    case "Tamil" => Tamil
    case "Thai" => Thai
    case "Turkish" => Turkish
    case "Ukranian" => Ukranian
    case "Urdu" => Urdu
    case "Vietnamese" => Vietnamese
    case _ => throw new IllegalArgumentException(s"'$name' is not a valid language name.")
  }

  @throws[IllegalArgumentException]
  def fromAbbreviation(abbreviation: String): Language = abbreviation match {
    case "sq" => Albanian
    case "ar" => Arabic
    case "hy" => Armenian
    case "az" => Azerbaijani
    case "eu" => Basque
    case "be" => Belarusian
    case "bn" => Bengali
    case "bs" => Bosnian
    case "bg" => Bulgarian
    case "my" => Burmese
    case "ca" => Catalan
    case "zh" => Chinese
    case "zh-cn" => ChineseMainland
    case "zh-tw" => ChineseTaiwan
    case "hr" => Croatian
    case "cs" => Czech
    case "da" => Danish
    case "nl" => Dutch
    case "en" => English
    case "eo" => Esperanto
    case "et" => Estonian
    case "fi" => Finnish
    case "fr" => French
    case "fr-ca" => FrenchCanada
    case "gl" => Galician
    case "de" => German
    case "ka" => Georgian
    case "el" => Greek
    case "he" => Hebrew
    case "hi" => Hindi
    case "hu" => Hungarian
    case "is" => Icelandic
    case "id" => Indonesian
    case "it" => Italian
    case "ga" => Irish
    case "ja" => Japanese
    case "kk" => Kazakh
    case "ko" => Korean
    case "ku" => Kurdish
    case "lt" => Lithuanian
    case "lv" => Latvian
    case "mk" => Macedonian
    case "ms" => Malay
    case "mr" => Marathi
    case "mn" => Mongolian
    case "nb" => Norwegian
    case "fa" => Persian
    case "pl" => Polish
    case "pt" => Portuguese
    case "pt-br" => PortugueseBrazil
    case "ro" => Romanian
    case "ru" => Russian
    case "sr" => Serbian
    case "sk" => Slovak
    case "sl" => Slovenian
    case "es" => Spanish
    case "sv" => Swedish
    case "ta" => Tamil
    case "th" => Thai
    case "tr" => Turkish
    case "uk" => Ukranian
    case "ur" => Urdu
    case "vi" => Vietnamese
    case _ => throw new IllegalArgumentException(s"'$abbreviation' is not a valid language abbreviation.")
  }
}
