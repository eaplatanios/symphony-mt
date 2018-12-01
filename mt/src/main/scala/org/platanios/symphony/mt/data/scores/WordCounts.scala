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

package org.platanios.symphony.mt.data.scores

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.data.{newReader, newWriter}
import org.platanios.symphony.mt.utilities.TrieWordCounter

import better.files._

import scala.collection.mutable
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
class WordCounts(val caseSensitive: Boolean = false) extends SummaryScore {
  protected val whitespaceRegex: Regex = "\\s+".r

  protected val counters: mutable.HashMap[Language, TrieWordCounter] = {
    mutable.HashMap.empty[Language, TrieWordCounter]
  }

  override def name: String = {
    if (caseSensitive) "cs-wc" else "wc"
  }

  override def processSentence(
      language: Language,
      sentence: String,
      requiredValues: Seq[Float],
      requiredSummaries: Seq[SummaryScore]
  ): Unit = {
    whitespaceRegex.split(sentence).foreach(word => {
      if (caseSensitive)
        counters.getOrElseUpdate(language, TrieWordCounter()).insertWord(word)
      else
        counters.getOrElseUpdate(language, TrieWordCounter()).insertWord(word.toLowerCase)
    })
  }

  override def resetState(): Unit = {
    counters.clear()
  }

  override def saveStateToFile(file: File): Unit = {
    val writer = newWriter(file)
    counters.foreach {
      case (language, counter) =>
        writer.write(s"${WordCounts.FILE_LANGUAGE_SEPARATOR}\n")
        writer.write(s"${language.name}\n")
        for ((count, word) <- counter.words().toSeq.sortBy(-_._1).distinct)
          writer.write(s"$word\t$count\n")
    }
    writer.flush()
    writer.close()
  }

  override def loadStateFromFile(file: File): Unit = {
    reset()
    var currentLanguage: Option[Language] = None
    newReader(file).lines().toAutoClosedIterator.foreach(line => {
      if (line == WordCounts.FILE_LANGUAGE_SEPARATOR) {
        currentLanguage = None
      } else {
        currentLanguage match {
          case None => currentLanguage = Some(Language.fromName(line))
          case Some(language) =>
            val lineParts = line.split('\t')
            counters.getOrElseUpdate(language, TrieWordCounter())
                .insertWordWithCount(word = lineParts(0), count = lineParts(1).toLong)
        }
      }
    })
  }

  def count(language: Language, word: String): Long = {
    counters(language)(word)
  }

  def totalCount(language: Language): Long = {
    counters(language).totalCount
  }
}

object WordCounts {
  private[WordCounts] val FILE_LANGUAGE_SEPARATOR: String = "-" * 100

  def apply(caseSensitive: Boolean = false): WordCounts = {
    new WordCounts(caseSensitive)
  }
}
