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
import org.platanios.symphony.mt.Language.Vietnamese
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

  override def process(file: File, language: Language): File = normalizeCorpus(file, language)

  def normalizedFile(originalFile: File): File = {
    val fileName = originalFile.nameWithoutExtension(includeAll = false) + s".normalized${originalFile.extension().get}"
    originalFile.sibling(fileName)
  }

  def normalize(sentence: String, language: Language): String = language match {
    case Vietnamese => replacementRegexSeq.foldLeft(sentence)((s, r) => r._1.replaceAllIn(s, r._2))
    case _ => sentence
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
