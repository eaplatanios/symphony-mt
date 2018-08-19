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

package org.platanios.symphony.mt.data.processors

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.Language._
import org.platanios.symphony.mt.data.{newReader, newWriter}

import better.files._
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
object Normalizer extends FileProcessor {
  private val logger = Logger(LoggerFactory.getLogger("Data / Normalizer"))

  private val replacementRegexSeq: Seq[(Regex, String)] = Seq(
    ("""À""".r, "À"), ("""Ã""".r, "Ã"), ("""Ả""".r, "Ả"), ("""Á""".r, "Á"), ("""Ạ""".r, "Ạ"),
    ("""Ằ""".r, "Ằ"), ("""Ẵ""".r, "Ẵ"), ("""Ẳ""".r, "Ẳ"), ("""Ắ""".r, "Ắ"), ("""Ặ""".r, "Ặ"),
    ("""Ầ""".r, "Ầ"), ("""Ẫ""".r, "Ẫ"), ("""Ẩ""".r, "Ẩ"), ("""Ấ""".r, "Ấ"), ("""Ậ""".r, "Ậ"),
    ("""Ỳ""".r, "Ỳ"), ("""Ỹ""".r, "Ỹ"), ("""Ỷ""".r, "Ỷ"), ("""Ý""".r, "Ý"), ("""Ỵ""".r, "Ỵ"),
    ("""Ì""".r, "Ì"), ("""Ĩ""".r, "Ĩ"), ("""Ỉ""".r, "Ỉ"), ("""Í""".r, "Í"), ("""Ị""".r, "Ị"),
    ("""Ù""".r, "Ù"), ("""Ũ""".r, "Ũ"), ("""Ủ""".r, "Ủ"), ("""Ú""".r, "Ú"), ("""Ụ""".r, "Ụ"),
    ("""Ừ""".r, "Ừ"), ("""Ữ""".r, "Ữ"), ("""Ử""".r, "Ử"), ("""Ứ""".r, "Ứ"), ("""Ự""".r, "Ự"),
    ("""È""".r, "È"), ("""Ẽ""".r, "Ẽ"), ("""Ẻ""".r, "Ẻ"), ("""É""".r, "É"), ("""Ẹ""".r, "Ẹ"),
    ("""Ề""".r, "Ề"), ("""Ễ""".r, "Ễ"), ("""Ể""".r, "Ể"), ("""Ế""".r, "Ế"), ("""Ệ""".r, "Ệ"),
    ("""Ò""".r, "Ò"), ("""Õ""".r, "Õ"), ("""Ỏ""".r, "Ỏ"), ("""Ó""".r, "Ó"), ("""Ọ""".r, "Ọ"),
    ("""Ờ""".r, "Ờ"), ("""Ỡ""".r, "Ỡ"), ("""Ở""".r, "Ở"), ("""Ớ""".r, "Ớ"), ("""Ợ""".r, "Ợ"),
    ("""Ồ""".r, "Ồ"), ("""Ỗ""".r, "Ỗ"), ("""Ổ""".r, "Ổ"), ("""Ố""".r, "Ố"), ("""Ộ""".r, "Ộ"),
    ("""ÒA""".r, "OÀ"), ("""ÕA""".r, "OÃ"), ("""ỎA""".r, "OẢ"), ("""ÓA""".r, "OÁ"), ("""ỌA""".r, "OẠ"),
    ("""ÒE""".r, "OÈ"), ("""ÕE""".r, "OẼ"), ("""ỎE""".r, "OẺ"), ("""ÓE""".r, "OÉ"), ("""ỌE""".r, "OẸ"),
    ("""ÙY""".r, "UỲ"), ("""ŨY""".r, "UỸ"), ("""ỦY""".r, "UỶ"), ("""ÚY""".r, "UÝ"), ("""ỤY""".r, "UỴ"),
    ("""à""".r, "à"), ("""ã""".r, "ã"), ("""ả""".r, "ả"), ("""á""".r, "á"), ("""ạ""".r, "ạ"),
    ("""ằ""".r, "ằ"), ("""ẵ""".r, "ẵ"), ("""ẳ""".r, "ẳ"), ("""ắ""".r, "ắ"), ("""ặ""".r, "ặ"),
    ("""ầ""".r, "ầ"), ("""ẫ""".r, "ẫ"), ("""ẩ""".r, "ẩ"), ("""ấ""".r, "ấ"), ("""ậ""".r, "ậ"),
    ("""ỳ""".r, "ỳ"), ("""ỹ""".r, "ỹ"), ("""ỷ""".r, "ỷ"), ("""ý""".r, "ý"), ("""ỵ""".r, "ỵ"),
    ("""ì""".r, "ì"), ("""ĩ""".r, "ĩ"), ("""ỉ""".r, "ỉ"), ("""í""".r, "í"), ("""ị""".r, "ị"),
    ("""ù""".r, "ù"), ("""ũ""".r, "ũ"), ("""ủ""".r, "ủ"), ("""ú""".r, "ú"), ("""ụ""".r, "ụ"),
    ("""ừ""".r, "ừ"), ("""ữ""".r, "ữ"), ("""ử""".r, "ử"), ("""ứ""".r, "ứ"), ("""ự""".r, "ự"),
    ("""è""".r, "è"), ("""ẽ""".r, "ẽ"), ("""ẻ""".r, "ẻ"), ("""é""".r, "é"), ("""ẹ""".r, "ẹ"),
    ("""ề""".r, "ề"), ("""ễ""".r, "ễ"), ("""ể""".r, "ể"), ("""ế""".r, "ế"), ("""ệ""".r, "ệ"),
    ("""ò""".r, "ò"), ("""õ""".r, "õ"), ("""ỏ""".r, "ỏ"), ("""ó""".r, "ó"), ("""ọ""".r, "ọ"),
    ("""ờ""".r, "ờ"), ("""ỡ""".r, "ỡ"), ("""ở""".r, "ở"), ("""ớ""".r, "ớ"), ("""ợ""".r, "ợ"),
    ("""ồ""".r, "ồ"), ("""ỗ""".r, "ỗ"), ("""ổ""".r, "ổ"), ("""ố""".r, "ố"), ("""ộ""".r, "ộ"),
    ("""òa""".r, "oà"), ("""õa""".r, "oã"), ("""ỏa""".r, "oả"), ("""óa""".r, "oá"), ("""ọa""".r, "oạ"),
    ("""òe""".r, "oè"), ("""õe""".r, "oẽ"), ("""ỏe""".r, "oẻ"), ("""óe""".r, "oé"), ("""ọe""".r, "oẹ"),
    ("""ùy""".r, "uỳ"), ("""ũy""".r, "uỹ"), ("""ủy""".r, "uỷ"), ("""úy""".r, "uý"), ("""ụy""".r, "uỵ"),
    ("""aó""".r, "áo"), ("""òA""".r, "oà"), ("""õA""".r, "oã"), ("""ỏA""".r, "oả"), ("""óA""".r, "oá"),
    ("""ọA""".r, "oạ"), ("""òE""".r, "oè"), ("""õE""".r, "oẽ"), ("""ỏE""".r, "oẻ"), ("""óE""".r, "oé"),
    ("""ọE""".r, "oẹ"), ("""ùY""".r, "uỳ"), ("""ũY""".r, "uỹ"), ("""ủY""".r, "uỷ"), ("""úY""".r, "uý"),
    ("""ụY""".r, "uỵ"), ("""Òa""".r, "Oà"), ("""Õa""".r, "Oã"), ("""Ỏa""".r, "Oả"), ("""Óa""".r, "Oá"),
    ("""Ọa""".r, "Oạ"), ("""Òe""".r, "Oè"), ("""Õe""".r, "Oẽ"), ("""Ỏe""".r, "Oẻ"), ("""Óe""".r, "Oé"),
    ("""Ọe""".r, "Oẹ"), ("""Ùy""".r, "Uỳ"), ("""Ũy""".r, "Uỹ"), ("""Ủy""".r, "Uỷ"), ("""Úy""".r, "Uý"),
    ("""Ụy""".r, "Uỵ"))

  private val whitespaceRegex    : Regex = """\s+""".r
  private val regexUnicodeNumber0: Regex = """０""".r
  private val regexUnicodeNumber2: Regex = """２""".r
  private val regexUnicodeNumber3: Regex = """３""".r
  private val regexUnicodeNumber4: Regex = """４""".r
  private val regexUnicodeNumber5: Regex = """５""".r
  private val regexUnicodeNumber6: Regex = """６""".r
  private val regexUnicodeNumber7: Regex = """７""".r
  private val regexUnicodeNumber8: Regex = """８""".r
  private val regexUnicodeNumber9: Regex = """９""".r
  private val unicodeRegex1      : Regex = """，|、""".r
  private val unicodeRegex2      : Regex = """。 *|． *""".r
  private val unicodeRegex3      : Regex = """‘|‚|’|''|´´|”|“|《|》|１|」|「""".r
  private val unicodeRegex4      : Regex = """∶|：""".r
  private val unicodeRegex5      : Regex = """？""".r
  private val unicodeRegex6      : Regex = """）""".r
  private val unicodeRegex7      : Regex = """！""".r
  private val unicodeRegex8      : Regex = """（""".r
  private val unicodeRegex9      : Regex = """；""".r
  private val unicodeRegex10     : Regex = """～""".r
  private val unicodeRegex11     : Regex = """’""".r
  private val unicodeRegex12     : Regex = """…|…""".r
  private val unicodeRegex13     : Regex = """〈""".r
  private val unicodeRegex14     : Regex = """〉""".r
  private val unicodeRegex15     : Regex = """【""".r
  private val unicodeRegex16     : Regex = """】""".r
  private val unicodeRegex17     : Regex = """％""".r
  private val regexEn            : Regex = """\"([,\.]+)""".r
  private val regex1             : Regex = """\(""".r
  private val regex2             : Regex = """\)""".r
  private val regex3             : Regex = """\) ([\.\!\:\?\;\,])""".r
  private val regex4             : Regex = """\( """.r
  private val regex5             : Regex = """ \)""".r
  private val regex6             : Regex = """(\d) \%""".r
  private val regex7             : Regex = """ :""".r
  private val regex8             : Regex = """ ;""".r
  private val regex9             : Regex = """\`""".r
  private val regex10            : Regex = """\'\'""".r
  private val regex11            : Regex = """„|“|”""".r
  private val regex12            : Regex = """–|━""".r
  private val regex13            : Regex = """—""".r
  private val regex14            : Regex = """´""".r
  private val regex15            : Regex = """([a-zA-Z])‘([a-zA-Z])""".r
  private val regex16            : Regex = """([a-zA-Z])’([a-zA-Z])""".r
  private val regex19            : Regex = """ « """.r
  private val regex20            : Regex = """« |«""".r
  private val regex21            : Regex = """ » """.r
  private val regex22            : Regex = """ »|»""".r
  private val regex23            : Regex = """ \%""".r
  private val regex24            : Regex = """nº """.r
  private val regex25            : Regex = """ ºC""".r
  private val regex26            : Regex = """ cm""".r
  private val regex27            : Regex = """ \?""".r
  private val regex28            : Regex = """ \!""".r
  private val regex29            : Regex = """,\"""".r
  private val regex30            : Regex = """(\.+)\"(\s*[^<])""".r
  private val regex31            : Regex = """(\d) (\d)""".r

  override def process(file: File, language: Language): File = normalizeCorpus(file, language)

  def normalizedFile(originalFile: File): File = {
    val fileName = originalFile.nameWithoutExtension(includeAll = false) + s".normalized${originalFile.extension().get}"
    originalFile.sibling(fileName)
  }

  def normalize(sentence: String, language: Language): String = {
    var normalized = language match {
      case Vietnamese => replacementRegexSeq.foldLeft(sentence)((s, r) => r._1.replaceAllIn(s, r._2))
      case _ => sentence
    }

    normalized = regexUnicodeNumber0.replaceAllIn(normalized, "0")
    normalized = regexUnicodeNumber2.replaceAllIn(normalized, "2")
    normalized = regexUnicodeNumber3.replaceAllIn(normalized, "3")
    normalized = regexUnicodeNumber4.replaceAllIn(normalized, "4")
    normalized = regexUnicodeNumber5.replaceAllIn(normalized, "5")
    normalized = regexUnicodeNumber6.replaceAllIn(normalized, "6")
    normalized = regexUnicodeNumber7.replaceAllIn(normalized, "7")
    normalized = regexUnicodeNumber8.replaceAllIn(normalized, "8")
    normalized = regexUnicodeNumber9.replaceAllIn(normalized, "9")
    normalized = unicodeRegex1.replaceAllIn(normalized, ",")
    normalized = unicodeRegex2.replaceAllIn(normalized, ". ")
    normalized = unicodeRegex3.replaceAllIn(normalized, "\"")
    normalized = unicodeRegex4.replaceAllIn(normalized, ":")
    normalized = unicodeRegex5.replaceAllIn(normalized, "?")
    normalized = unicodeRegex6.replaceAllIn(normalized, ")")
    normalized = unicodeRegex7.replaceAllIn(normalized, "!")
    normalized = unicodeRegex8.replaceAllIn(normalized, "(")
    normalized = unicodeRegex9.replaceAllIn(normalized, ";")
    normalized = unicodeRegex10.replaceAllIn(normalized, "~")
    normalized = unicodeRegex11.replaceAllIn(normalized, "'")
    normalized = unicodeRegex12.replaceAllIn(normalized, "...")
    normalized = unicodeRegex13.replaceAllIn(normalized, "<")
    normalized = unicodeRegex14.replaceAllIn(normalized, ">")
    normalized = unicodeRegex15.replaceAllIn(normalized, "[")
    normalized = unicodeRegex16.replaceAllIn(normalized, "]")
    normalized = unicodeRegex17.replaceAllIn(normalized, "%")
    normalized = regex1.replaceAllIn(normalized, " (")
    normalized = regex2.replaceAllIn(normalized, ") ")
    normalized = whitespaceRegex.replaceAllIn(normalized, " ")
    normalized = regex3.replaceAllIn(normalized, ")$1")
    normalized = regex4.replaceAllIn(normalized, "(")
    normalized = regex5.replaceAllIn(normalized, ")")
    normalized = regex6.replaceAllIn(normalized, "$1%")
    normalized = regex7.replaceAllIn(normalized, ":")
    normalized = regex8.replaceAllIn(normalized, ";")
    normalized = regex9.replaceAllIn(normalized, "'")
    normalized = regex10.replaceAllIn(normalized, " \" ")
    normalized = regex11.replaceAllIn(normalized, "\"")
    normalized = regex12.replaceAllIn(normalized, "-")
    normalized = regex13.replaceAllIn(normalized, " - ")
    normalized = whitespaceRegex.replaceAllIn(normalized, " ")
    normalized = regex14.replaceAllIn(normalized, "'")
    normalized = regex15.replaceAllIn(normalized, "$1'$2")
    normalized = regex16.replaceAllIn(normalized, "$1'$2")
    normalized = regex19.replaceAllIn(normalized, " \"")
    normalized = regex20.replaceAllIn(normalized, "\"")
    normalized = regex21.replaceAllIn(normalized, "\" ")
    normalized = regex22.replaceAllIn(normalized, "\"")
    normalized = regex23.replaceAllIn(normalized, "%")
    normalized = regex24.replaceAllIn(normalized, "nº")
    normalized = regex25.replaceAllIn(normalized, "ºC")
    normalized = regex26.replaceAllIn(normalized, "cm")
    normalized = regex27.replaceAllIn(normalized, "?")
    normalized = regex28.replaceAllIn(normalized, "!")
    normalized = regex7.replaceAllIn(normalized, ":")
    normalized = regex8.replaceAllIn(normalized, ";")
    normalized = whitespaceRegex.replaceAllIn(normalized, " ")
    if (language == English) {
      normalized = regexEn.replaceAllIn(normalized, "$1\"")
    } else if (language != Czech) {
      normalized = regex29.replaceAllIn(normalized, "\",")
      normalized = regex30.replaceAllIn(normalized, "\"$1$2")
    }

    if (language == German || language == Spanish || language == Czech || language == French)
      normalized = regex31.replaceAllIn(normalized, "$1,$2")
    else
      normalized = regex31.replaceAllIn(normalized, "$1.$2")

    normalized
  }

  def normalizeCorpus(file: File, language: Language): File = {
    val normalized = normalizedFile(file)
    if (normalized.notExists) {
      logger.info(s"Normalizing '$file'.")
      val tokenizedWriter = newWriter(normalized)
      newReader(file).lines().toAutoClosedIterator.foreach(sentence => {
        tokenizedWriter.write(s"${normalize(sentence, language)}\n")
      })
      tokenizedWriter.flush()
      tokenizedWriter.close()
      logger.info(s"Created normalized file '$normalized'.")
    }
    normalized
  }
}
