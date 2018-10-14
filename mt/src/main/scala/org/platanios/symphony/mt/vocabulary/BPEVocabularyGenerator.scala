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

package org.platanios.symphony.mt.vocabulary

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.data.{newReader, newWriter}
import org.platanios.symphony.mt.utilities.{MutableFile, PriorityCounter, TrieWordCounter}

import better.files._
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.io.BufferedWriter

import scala.collection.mutable
import scala.collection.parallel.mutable.ParMap
import scala.util.matching.Regex

/** Uses byte pair encoding (BPE) to learn a variable-length encoding of the vocabulary in a text.
  * Unlike the original BPE, it does not compress the plain text, but can be used to reduce the vocabulary of a text to
  * a configurable number of symbols, with only a small increase in the number of tokens.
  *
  * '''Reference:''' Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words
  * with Subword Units. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics
  * (ACL 2016). Berlin, Germany.
  *
  * '''NOTE:''' Our implementation is based upon the one of the original paper authors, which can be found at
  * [https://github.com/rsennrich/subword-nmt](https://github.com/rsennrich/subword-nmt).
  *
  * @param  numMergeOps     Number of BPE merge operations to learn when generating a new BPE vocabulary.
  * @param  separator       Separator symbol appended to all inter-word symbols while encoding sentences. This allows
  *                         for decoding BPE-encoded sentences, after a translation is done.
  * @param  glossary
  * @param  caseSensitive
  * @param  countThreshold  Symbols pairs which appears less than `countThreshold` times will be ignored.
  * @param  replaceExisting If `true`, existing vocabulary files will be replaced, if found.
  * @param  bufferSize      Buffer size to use while reading and writing files.
  *
  * @author Emmanouil Antonios Platanios
  */
class BPEVocabularyGenerator protected (
    val numMergeOps: Int = 32000,
    val separator: String = "@@",
    val glossary: Set[String] = BPEVocabularyGenerator.DEFAULT_GLOSSARY,
    val caseSensitive: Boolean = true,
    val countThreshold: Int = -1,
    val replaceExisting: Boolean = false,
    val bufferSize: Int = 8192
) extends VocabularyGenerator {
  protected val glossaryRegex: Regex = BPEVocabularyGenerator.glossaryRegex(glossary)

  protected val mergePairs        : mutable.Map[Seq[Language], Map[(String, String), Int]]    = mutable.Map.empty
  protected val reversedMergePairs: mutable.Map[Seq[Language], Map[String, (String, String)]] = mutable.Map.empty
  protected val vocabularies      : mutable.Map[Seq[Language], Set[String]]                   = mutable.Map.empty

  protected def suffix(languages: Seq[Language]): String = {
    languages.map(_.abbreviation).sorted.mkString(".")
  }

  protected def mergePairsFilename(languages: Seq[Language]): String = {
    s"merge_pairs.bpe.$numMergeOps.${suffix(languages)}"
  }

  /** Returns the vocabulary file name that this generator uses / will use.
    *
    * @param  languages Languages for which a vocabulary will be generated.
    * @return Vocabulary file name.
    */
  override def filename(languages: Seq[Language]): String = {
    s"vocab.bpe.$numMergeOps.${suffix(languages)}"
  }

  /** Generates/Replaces a vocabulary file given a sequence of tokenized text files.
    *
    * @param  languages      Languages for which a merged vocabulary will be generated.
    * @param  tokenizedFiles Tokenized text files to use for generating the vocabulary file.
    * @param  vocabDir       Directory in which to save the generated vocabulary files.
    * @return The generated/replaced vocabulary file.
    */
  override protected def generate(
      languages: Seq[Language],
      tokenizedFiles: Seq[MutableFile],
      vocabDir: File
  ): File = {
    if (!mergePairs.contains(languages)) {
      mergePairs += languages -> Map.empty
      reversedMergePairs += languages -> Map.empty
      vocabularies += languages -> Set.empty
    }

    // We first generate the merge pairs, if necessary.
    val mergePairsFile = vocabDir / mergePairsFilename(languages)
    if (mergePairsFile.exists && !replaceExisting) {
      BPEVocabularyGenerator.logger.info(
        s"Loading existing BPE coding for ${languages.mkString(", ")}: $mergePairsFile.")
      initializeMergePairs(languages, vocabDir)
    } else {
      BPEVocabularyGenerator.logger.info(s"Learning BPE coding for ${languages.mkString(", ")}.")
      mergePairsFile.parent.createDirectories()
      val mergePairsWriter = newWriter(mergePairsFile)
      val tokens = mutable.ArrayBuffer(tokenizedFiles.map(_.get).toIterator.flatMap(file => {
        newReader(file).lines().toAutoClosedIterator
            .flatMap(BPEVocabularyGenerator.whitespaceRegex.split)
      }).foldLeft(TrieWordCounter())((counter, word) => {
        counter.insertWord(word)
        counter
      }).words().map(p => (p._1, {
        val parts = {
          if (caseSensitive)
            p._2.toCharArray.map(_.toString)
          else
            p._2.toLowerCase().toCharArray.map(_.toString)
        }
        parts.update(parts.length - 1, parts.last + BPEVocabularyGenerator.END_OF_WORD_SYMBOL)
        parts.toSeq
      })).toSeq: _*)

      val pairStatistics = BPEVocabularyGenerator.computePairStatistics(tokens)
      val counts = pairStatistics.counts
      val indices = pairStatistics.indices

      var continue = true
      var currentSymbol = 0
      var progressLogTime = System.currentTimeMillis

      while (currentSymbol < numMergeOps && continue) {
        val time = System.currentTimeMillis
        if (time - progressLogTime >= 1e4) {
          val numBars = Math.floorDiv(10 * currentSymbol, numMergeOps)
          BPEVocabularyGenerator.logger.info(
            s"│${"═" * numBars}${" " * (10 - numBars)}│ " +
                s"%${numMergeOps.toString.length}s / $numMergeOps BPE symbols processed.".format(currentSymbol))
          progressLogTime = time
        }

        val mostFrequent = if (counts.nonEmpty) counts.dequeueMax() else null

        if (mostFrequent._1 < countThreshold) {
          BPEVocabularyGenerator.logger.info(
            s"No pair has frequency higher than $countThreshold. Stopping the byte pair encoding (BPE) iteration.")
          continue = false
        } else {
          if (!mergePairs(languages).contains(mostFrequent._2)) {
            mergePairs(languages) += mostFrequent._2 -> currentSymbol
            reversedMergePairs(languages) += mostFrequent._2._1 + mostFrequent._2._2 -> mostFrequent._2
            mergePairsWriter.write(s"${mostFrequent._2._1}\t${mostFrequent._2._2}\n")
          }
          val changes = BPEVocabularyGenerator.replacePair(mostFrequent._2, tokens, indices)
          BPEVocabularyGenerator.updatePairStatistics(mostFrequent._2, changes, counts, indices)
        }

        currentSymbol += 1
      }

      mergePairsWriter.flush()
      mergePairsWriter.close()

      BPEVocabularyGenerator.logger.info(
        s"│${"═" * 10}│ %${numMergeOps.toString.length}s / $numMergeOps BPE symbols processed.".format(currentSymbol))
      BPEVocabularyGenerator.logger.info(s"Learned BPE coding for ${languages.mkString(", ")}: $mergePairsFile.")
    }

    // We then generate the vocabulary, if necessary.
    val vocabFile = vocabDir / filename(languages)
    val vocabWriter = {
      if (vocabFile.exists && !replaceExisting) {
        BPEVocabularyGenerator.logger.info(s"Vocabulary for ${languages.mkString(", ")} already exists: $vocabFile.")
        initializeVocabularies(languages, vocabDir)
        None
      } else {
        BPEVocabularyGenerator.logger.info(s"Generating vocabulary file for ${languages.mkString(", ")}.")
        vocabFile.parent.createDirectories()
        Some(newWriter(vocabFile))
      }
    }

    // Irrespective of whether a new vocabulary is being generated, or an existing one was loaded, we also convert the
    // provided tokenized files to their encoded equivalent.
    var fileWriters = Seq.empty[BufferedWriter]
    val tokens = tokenizedFiles.flatMap(mutableFile => {
      val oldFile = mutableFile.get
      val file = oldFile.sibling(
        s"${oldFile.nameWithoutExtension(includeAll = false)}" +
            s".bpe.$numMergeOps.${languages.map(_.abbreviation).sorted.mkString(".")}" +
            s".${oldFile.extension(includeDot = false, includeAll = false).get}")
      mutableFile.set(file)
      if (replaceExisting || file.notExists) {
        BPEVocabularyGenerator.logger.info(s"Applying BPE coding to file: $oldFile.")
        val fileWriter = if (replaceExisting || file.notExists) Some(newWriter(file)) else None
        val cache = mutable.Map.empty[String, Seq[String]]
        val tokens = newReader(oldFile).lines().toAutoClosedIterator
            .filter(_.length > 0)
            .flatMap(line => {
              var sentence = BPEVocabularyGenerator.whitespaceRegex.split(line)
              sentence = encodeSentence(languages, sentence, cache).toArray
              if (sentence.nonEmpty)
                fileWriter.foreach(_.write(s"${sentence.mkString(" ")}\n"))
              if (replaceExisting || vocabWriter.isDefined)
                sentence.toIterator
              else
                Iterator.empty
            })
        fileWriter.foreach(fileWriters :+= _)
        tokens
      } else if (vocabWriter.isDefined) {
        newReader(file).lines().toAutoClosedIterator
            .flatMap(line => BPEVocabularyGenerator.whitespaceRegex.split(line))
      } else {
        Iterator.empty
      }
    })

    vocabWriter.foreach(writer => {
      tokens.foldLeft(TrieWordCounter())((counter, word) => {
        counter.insertWord(word.trim)
        counter
      }).words()
          .toSeq
          .sortBy(-_._1)
          .map(_._2)
          .distinct
          .foreach(word => {
            vocabularies(languages) += word
            writer.write(word + "\n")
          })
      writer.flush()
      writer.close()
      BPEVocabularyGenerator.logger.info(s"Generated vocabulary file for ${languages.mkString(", ")}.")
    })

    fileWriters.foreach(fileWriter => {
      fileWriter.flush()
      fileWriter.close()
    })

    BPEVocabularyGenerator.logger.info(s"Applied BPE coding to all provided files for ${languages.mkString(", ")}.")

    vocabFile
  }

  /** Initializes the merge pairs of this BPE generators from an existing file.
    *
    * @param  languages Languages for which a vocabulary has been generated.
    * @param  vocabDir  Directory in which the generated vocabulary file and any other relevant files have been saved.
    */
  protected def initializeMergePairs(languages: Seq[Language], vocabDir: File): Unit = {
    val mergePairsFile = vocabDir / mergePairsFilename(languages)
    mergePairs += languages -> newReader(mergePairsFile).lines().toAutoClosedIterator
        .filter(_ != "")
        .map(l => {
          val parts = l.split("\t")
          (parts(0), parts(1))
        }).zipWithIndex.toMap
    reversedMergePairs += languages -> mergePairs(languages).toSeq.map(p => p._1._1 + p._1._2 -> p._1).toMap
  }

  /** Initializes the vocabularies of this BPE generators from an existing file.
    *
    * @param  languages Languages for which a vocabulary has been generated.
    * @param  vocabDir  Directory in which the generated vocabulary file and any other relevant files have been saved.
    */
  protected def initializeVocabularies(languages: Seq[Language], vocabDir: File): Unit = {
    val vocabFile = vocabDir / filename(languages)
    vocabularies += languages -> newReader(vocabFile).lines().toAutoClosedIterator
        .filter(_ != "")
        .toSet
  }

  /** Returns a vocabulary for the specified languages, ready to be used by machine translation models.
    *
    * @param  languages Languages for which to return a vocabulary.
    * @param  vocabDir  Directory in which the generated vocabulary file and any other relevant files have been saved.
    * @return Created vocabulary.
    */
  override protected def getVocabulary(languages: Seq[Language], vocabDir: File): Vocabulary = {
    CodedVocabulary(vocabDir / filename(languages), s => encodeSentence(languages, s), decodeSentence)
  }

  /** Encodes the provided sentence to a sequence of BPE coded words.
    *
    * @param  languages Languages in which the sentence is written.
    * @param  sentence  Sentence to encode as a sequence of words.
    * @param  cache     Optional cache of already encoded words, used to speed up the encoding process.
    * @return Encoded sentence as a sequence of BPE coded words.
    */
  def encodeSentence(
      languages: Seq[Language],
      sentence: Seq[String],
      cache: mutable.Map[String, Seq[String]] = mutable.Map.empty
  ): Seq[String] = {
    // TODO: Add support for glossaries (i.e., words that will be encoded with the identity function.
    sentence.flatMap(word => {
      var parts = encodeWord(languages, word, cache)
      var i = 0
      while (i < parts.length - 1) {
        parts = parts.updated(i, parts(i) + separator)
        i += 1
      }
      parts
    })
  }

  /** Encodes the provided word to a sequence of BPE coded words.
    *
    * @param  languages Languages in which the word is written.
    * @param  word      Word to encode.
    * @param  cache     Optional cache of already encoded words, used to speed up the encoding process.
    * @return Encoded word as a sequence of BPE coded words.
    */
  def encodeWord(
      languages: Seq[Language],
      word: String,
      cache: mutable.Map[String, Seq[String]] = mutable.Map.empty
  ): Seq[String] = {
    cache.getOrElseUpdate(word, {
      var wordParts = BPEVocabularyGenerator.splitWithDelimiters(word, glossaryRegex, keepEmpty = false)
      if (wordParts.length < 2) {
        wordParts
      } else {
        wordParts = wordParts.updated(
          wordParts.length - 1, wordParts.last + BPEVocabularyGenerator.END_OF_WORD_SYMBOL)
        var pairs = {
          if (caseSensitive)
            wordParts.sliding(2).map(p => (p(0), p(1))).toArray
          else
            wordParts.sliding(2).map(p => (p(0).toLowerCase(), p(1).toLowerCase())).toArray
        }
        var continue = true
        while (pairs.nonEmpty && continue) {
          val pair = pairs.map(p => (p, mergePairs(languages).get(p))).minBy(_._2.getOrElse(Int.MaxValue))
          if (pair._2.isEmpty) {
            continue = false
          } else {
            wordParts = BPEVocabularyGenerator.replacePair(pair._1, wordParts, caseSensitive)
            pairs = {
              if (wordParts.length < 2)
                Array.empty
              else if (caseSensitive)
                wordParts.sliding(2).map(p => (p(0), p(1))).toArray
              else
                wordParts.sliding(2).map(p => (p(0).toLowerCase(), p(1).toLowerCase())).toArray
            }
          }
        }

        // Remove end-of-word symbols.
        if (wordParts.last.endsWith(BPEVocabularyGenerator.END_OF_WORD_SYMBOL))
          wordParts = wordParts.updated(wordParts.length - 1, wordParts.last
              .dropRight(BPEVocabularyGenerator.END_OF_WORD_SYMBOL.length))

        // Check if the new words parts are in the vocabulary, and backtrack if necessary.
        wordParts = BPEVocabularyGenerator.checkVocabularyAndSplit(
          wordParts, reversedMergePairs(languages), vocabularies(languages), separator)

        wordParts
      }
    })
  }

  /** Decodes the provided sentence to a sequence of words (before the BPE encoding was applied).
    *
    * @param  sentence Sentence to decode as a sequence of BPE coded words.
    * @return Decoded sentence as a sequence of words.
    */
  def decodeSentence(sentence: Seq[String]): Seq[String] = {
    val decodedSentence = mutable.ArrayBuffer.empty[String]
    var i = 0
    var j = 0
    while (i < sentence.length) {
      if (j >= decodedSentence.length)
        decodedSentence += ""
      if (sentence(i).endsWith(separator)) {
        decodedSentence(j) += sentence(i).dropRight(separator.length)
      } else {
        decodedSentence(j) += sentence(i)
        j += 1
      }
      i += 1
    }
    decodedSentence
  }

  override def toString: String = {
    if (countThreshold > 0)
      s"bpe-$numMergeOps-$countThreshold"
    else
      s"bpe-$numMergeOps"
  }
}

object BPEVocabularyGenerator {
  private[BPEVocabularyGenerator] val logger = Logger(LoggerFactory.getLogger("Vocabulary / BPE Generator"))

  def apply(
      numMergeOps: Int = 32000,
      separator: String = "@@",
      glossary: Set[String] = DEFAULT_GLOSSARY,
      caseSensitive: Boolean = true,
      countThreshold: Int = -1,
      replaceExisting: Boolean = false,
      bufferSize: Int = 8192
  ): BPEVocabularyGenerator = {
    new BPEVocabularyGenerator(
      numMergeOps, separator, glossary, caseSensitive, countThreshold, replaceExisting, bufferSize)
  }

  /** End-of-word symbol used by the BPE vocabulary generator. */
  val END_OF_WORD_SYMBOL: String = "</w>"

  /** Regular expression used for tokenizing sentences. */
  private[BPEVocabularyGenerator] val whitespaceRegex: Regex = "\\s+".r

  /** Default glossary to use. */
  private[BPEVocabularyGenerator] val DEFAULT_GLOSSARY: Set[String] = Set(
    "e.g", "i.e", "&amp;", "&#124;", "&lt;", "&gt;", "&apos;", "&quot;", "&#91;", "&#93;")

  private[BPEVocabularyGenerator] def glossaryRegex(glossary: Set[String]): Regex = {
    s"(?:${glossary.mkString("|")})|(?!${glossary.mkString("|")})".r
  }

  private[BPEVocabularyGenerator] def splitWithDelimiters(
      string: String,
      regex: Regex,
      keepEmpty: Boolean = false
  ): Seq[String] = {
    val parts = mutable.ArrayBuffer.empty[String]
    parts.sizeHint(string.length)
    val p = regex.pattern
    val m = p.matcher(string)
    var lastEnd = 0
    while (m.find) {
      val start = m.start
      if (lastEnd != start)
        parts += string.substring(lastEnd, start)
      if (keepEmpty || m.group.length > 0)
        parts += m.group
      lastEnd = m.end
    }
    if (lastEnd != string.length)
      parts += string.substring(lastEnd)
    parts
  }

  private[BPEVocabularyGenerator] case class PairStatistics(
      counts: PriorityCounter[(String, String)],
      indices: ParMap[(String, String), mutable.LongMap[Long]])

  private[BPEVocabularyGenerator] case class Change(
      index: Int,
      word: Seq[String],
      newWord: Seq[String],
      count: Long)

  private[BPEVocabularyGenerator] def updateIndices(
      indices: ParMap[(String, String), mutable.LongMap[Long]],
      pair: (String, String),
      index: Int,
      increment: Long
  ): Unit = {
    if (!indices.contains(pair))
      indices.put(pair, mutable.LongMap.empty[Long])
    if (!indices(pair).contains(index))
      indices(pair).put(index, increment)
    else
      indices(pair).put(index, indices(pair)(index) + increment)
  }

  /** Computes the pair statistics for the provided vocabulary of words.
    *
    * @param  words Vocabulary of words for which to compute statistics (each tuple contains a count and a word).
    * @return Computed statistics.
    */
  private[BPEVocabularyGenerator] def computePairStatistics(
      words: Seq[(Long, Seq[String])]
  ): PairStatistics = {
    val counts = PriorityCounter[(String, String)]()
    val indices = ParMap.empty[(String, String), mutable.LongMap[Long]]
    words.zipWithIndex.filter(_._1._2.length > 1).foreach {
      case ((count, symbols), index) =>
        symbols.sliding(2).foreach(s => {
          val pair = (s(0), s(1))
          counts.add(pair, count)
          updateIndices(indices, pair, index, 1)
        })
    }
    PairStatistics(counts, indices)
  }

  /** Replaces all occurrences of the provided symbol pair in `words` with the joined symbol.
    *
    * '''NOTE:''' This method mutates the provided `words` sequence.
    *
    * @param  pair    Symbol pair to replace in all sequences contained in `words`.
    * @param  words   Sequence of words treated as the current vocabulary (along with their counts).
    * @param  indices Map containing the indices where each symbol pair is found in `words`, along with their
    *                 corresponding counts.
    * @return Collection of changes made to `words`.
    */
  private[BPEVocabularyGenerator] def replacePair(
      pair: (String, String),
      words: mutable.Seq[(Long, Seq[String])],
      indices: ParMap[(String, String), mutable.LongMap[Long]]
  ): Seq[Change] = {
    indices(pair).toSeq.filter(_._2 >= 1).map(_._1.toInt).map(index => {
      val (count, word) = words(index)
      val newWord = replacePair(pair, word, caseSensitive = false)
      words.update(index, (count, newWord))
      Change(index, word, newWord, count)
    }).seq
  }

  /** Replaces all occurrences of the provided symbol pair in `word` with the joined symbol.
    *
    * @param  pair          Symbol pair to replace in `word`.
    * @param  word          Word as a sequence of symbols.
    * @param  caseSensitive Boolean indicating whether to be case-sensitive.
    * @return New word with `pair` replaced in `word`.
    */
  @inline
  private[BPEVocabularyGenerator] final def replacePair(
      pair: (String, String),
      word: Seq[String],
      caseSensitive: Boolean
  ): Seq[String] = {
    val newWord = mutable.ListBuffer.empty[String]
    var j = 0
    while (j < word.length - 1) {
      val joinedPair = word(j) + word(j + 1)
      (word(j), word(j + 1)) match {
        case p if caseSensitive && p == pair => newWord += joinedPair; j += 2
        case p if (p._1.toLowerCase(), p._2.toLowerCase()) == pair => newWord += joinedPair; j += 2
        case _ => newWord += word(j); j += 1
      }
    }
    if (j == word.length - 1)
      newWord += word(j)
    newWord
  }

  /** Minimally updates the symbol pair statistics, based on the provided list of changes.
    *
    * This method takes advantage of the fact that if we merge a pair of symbols, only pairs that overlap with
    * occurrences of this pair are affected and need to be updated.
    *
    * @param  pair    Symbol pair that was merged.
    * @param  changes Changes in the vocabulary of words resulting from the merge.
    * @param  counts  Symbol pair counts that will be updated.
    * @param  indices Map that will be updated, containing the indices where each symbol pair is found in `words`,
    *                 along with their corresponding counts.
    */
  private[BPEVocabularyGenerator] def updatePairStatistics(
      pair: (String, String),
      changes: Seq[Change],
      counts: PriorityCounter[(String, String)],
      indices: ParMap[(String, String), mutable.LongMap[Long]]
  ): Unit = {
    val joinedPair = pair._1 + pair._2

    // TODO: counts -= pair
    counts.update(pair, 0)
    indices(pair).clear()
    changes.foreach(change => {
      // Find all instances of the pair, and update the corresponding statistics.
      var i = 0
      while (i < change.word.length - 1) {
        (change.word(i), change.word(i + 1)) match {
          case p if p == pair =>
            // Assuming a symbol sequence "A B C", if "B C" is merged, we reduce the frequency of "A B".
            if (i > 0) {
              val prevPair = (change.word(i - 1), change.word(i))
              counts.add(prevPair, -change.count)
              updateIndices(indices, prevPair, change.index, -1)
            }
            // Assuming a symbol sequence "A B C B", if "B C" is merged, we reduce the frequency of "C B". However, we
            // skip this if the sequence is "A B C B C", because the frequency of "C B" will have already been reduced
            // by the previous code block.
            if (i < change.word.length - 2 &&
                (change.word(i + 2) != pair._1 || i >= change.word.length - 3 || change.word(i + 3) != pair._2)) {
              val nextPair = (change.word(i + 1), change.word(i + 2))
              counts.add(nextPair, -change.count)
              updateIndices(indices, nextPair, change.index, -1)
            }
            i += 2
          case _ => i += 1
        }
      }

      // Find all instances of the joined pair, and update the corresponding statistics.
      i = 0
      while (i < change.newWord.length) {
        change.newWord(i) match {
          case w if w == joinedPair =>
            // Assuming a symbol sequence "A BC D", if "B C" is merged, we increase the frequency of "A BC".
            if (i > 0) {
              val prevPair = (change.newWord(i - 1), change.newWord(i))
              counts.add(prevPair, change.count)
              updateIndices(indices, prevPair, change.index, 1)
            }
            // Assuming a symbol sequence "A BC B", if "B C" is merged, we increase the frequency of "BC B". However, we
            // skip this if the sequence is "A BC BC", because the count of "BC BC" will have already been incremented
            // by the previous code block.
            if (i < change.newWord.length - 1 && change.newWord(i + 1) != joinedPair) {
              val nextPair = (change.newWord(i), change.newWord(i + 1))
              counts.add(nextPair, change.count)
              updateIndices(indices, nextPair, change.index, -1)
            }
            i += 1
          case _ => i += 1
        }
      }
    })
  }

  /** Recursively splits `wordPart` into smaller units (by reversing BPE merges) until all units are either in the
    * provided vocabulary, or cannot be split further.
    *
    * @param  wordPart           Word part that needs to be split.
    * @param  reversedMergePairs Reversed symbol pairs merge map.
    * @param  vocabulary         Vocabulary of valid words.
    * @param  separator          Separator used to denote where a word is split.
    * @param  isLast             Boolean value indicating whether `wordPart` is the last part of the word being split.
    * @return `wordPart` split segments.
    */
  private[BPEVocabularyGenerator] def splitRecursively(
      wordPart: String,
      reversedMergePairs: Map[String, (String, String)],
      vocabulary: Set[String],
      separator: String,
      isLast: Boolean = false
  ): Seq[String] = {
    val pair = {
      if (isLast) {
        reversedMergePairs.get(wordPart + END_OF_WORD_SYMBOL)
            .map(p => (p._1, p._2.dropRight(END_OF_WORD_SYMBOL.length)))
      } else {
        reversedMergePairs.get(wordPart)
      }
    }

    // TODO: !!! What about the case-insensitive case?

    pair match {
      case None => Seq(wordPart)
      case Some((left, right)) =>

        // We first go through the left parts.
        val leftParts = {
          if (vocabulary.contains(left + separator))
            Seq(left + separator)
          else
            splitRecursively(left, reversedMergePairs, vocabulary, separator, isLast = false)
        }

        // We then go through the right parts.
        val rightParts = {
          if (isLast && vocabulary.contains(right))
            Seq(right)
          else if (!isLast && vocabulary.contains(right + separator))
            Seq(right + separator)
          else
            splitRecursively(right, reversedMergePairs, vocabulary, separator, isLast = isLast)
        }

        leftParts ++ rightParts
    }
  }

  /** Checks for each part in `wordParts` if it is in the provided vocabulary, and segments out-of-vocabulary parts into
    * smaller units by reversing BPE merge operations.
    *
    * @param  wordParts          Word parts to check and split if necessary.
    * @param  reversedMergePairs Reversed symbol pairs merge map.
    * @param  vocabulary         Vocabulary of valid words.
    * @param  separator          Separator used to denote where a word is split.
    * @return New sequence of word parts that represents the same word and may be longer than the provided one.
    */
  private[BPEVocabularyGenerator] def checkVocabularyAndSplit(
      wordParts: Seq[String],
      reversedMergePairs: Map[String, (String, String)],
      vocabulary: Set[String],
      separator: String
  ): Seq[String] = {
    if (vocabulary.isEmpty) {
      wordParts
    } else {
      wordParts.zipWithIndex.flatMap {
        case (part, index) if index < wordParts.length - 1 && vocabulary.contains(part + separator) =>
          Seq(part + separator)
        case (part, index) if index < wordParts.length - 1 =>
          splitRecursively(part, reversedMergePairs, vocabulary, separator, isLast = false)
        case (part, _) if vocabulary.contains(part) => Seq(part)
        case (part, _) => splitRecursively(part, reversedMergePairs, vocabulary, separator, isLast = true)
      }
    }
  }
}
