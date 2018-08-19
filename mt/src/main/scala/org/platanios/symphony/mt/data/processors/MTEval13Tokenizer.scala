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

import better.files.File

import scala.util.matching.Regex

/** Tokenizer used by the official `mteval-v13a.perl` script.
  *
  * The code in this class is a Scala translation of the
  * [original script code](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v13a.pl).
  *
  * @param  preserveCase If `true`, case will be preserved.
  *
  * @author Emmanouil Antonios Platanios
  */
case class MTEval13Tokenizer(
    preserveCase: Boolean = true
) extends Tokenizer {
  private val skippedRegex                         : Regex = """<skipped>""".r
  private val endOfLineHyphenationRegex            : Regex = """-\n""".r
  private val joinLinesRegex                       : Regex = """\n""".r
  private val sgmlQuoteRegex                       : Regex = """&quot;""".r
  private val sgmlAmpersandRegex                   : Regex = """&amp;""".r
  private val sgmlLtRegex                          : Regex = """&lt;""".r
  private val sgmlGtRegex                          : Regex = """&gt;""".r
  private val punctuationRegex                     : Regex = """([\{-\~\[-\` -\&\(-\+\:-\@\/])""".r
  private val periodCommaUnlessPrecededByDigitRegex: Regex = """([^0-9])([\.,])""".r
  private val periodCommaUnlessFollowedByDigitRegex: Regex = """([\.,])([^0-9])""".r
  private val dashPrecededByDigitRegex             : Regex = """([0-9])(-)""".r
  private val whitespaceRegex                      : Regex = """\s+""".r
  private val leadingWhitespaceRegex               : Regex = """^\s+""".r
  private val trailingWhitespaceRegex              : Regex = """\s+$""".r

  override def tokenizedFile(originalFile: File): File = {
    val fileName = originalFile.nameWithoutExtension(includeAll = false)
    originalFile.sibling(fileName + s".tok:mteval13${originalFile.extension().get}")
  }

  override def tokenize(sentence: String, language: Language): String = {
    var tokenized = sentence.trim

    // Language-independent part.
    tokenized = skippedRegex.replaceAllIn(tokenized, "")
    tokenized = endOfLineHyphenationRegex.replaceAllIn(tokenized, "")
    tokenized = joinLinesRegex.replaceAllIn(tokenized, " ")
    tokenized = sgmlQuoteRegex.replaceAllIn(tokenized, "\"")
    tokenized = sgmlAmpersandRegex.replaceAllIn(tokenized, "&")
    tokenized = sgmlLtRegex.replaceAllIn(tokenized, "<")
    tokenized = sgmlGtRegex.replaceAllIn(tokenized, ">")

    // Language-dependent part (assuming Western languages).
    tokenized = s" $tokenized "
    if (!preserveCase)
      tokenized = tokenized.toLowerCase
    tokenized = punctuationRegex.replaceAllIn(tokenized, " $1 ")
    tokenized = periodCommaUnlessPrecededByDigitRegex.replaceAllIn(tokenized, "$1 $2 ")
    tokenized = periodCommaUnlessFollowedByDigitRegex.replaceAllIn(tokenized, " $1 $2")
    tokenized = dashPrecededByDigitRegex.replaceAllIn(tokenized, "$1 $2 ")
    tokenized = whitespaceRegex.replaceAllIn(tokenized, " ")
    tokenized = leadingWhitespaceRegex.replaceAllIn(tokenized, "")
    tokenized = trailingWhitespaceRegex.replaceAllIn(tokenized, "")

    tokenized
  }

  override def toString: String = "t:mteval13"
}
