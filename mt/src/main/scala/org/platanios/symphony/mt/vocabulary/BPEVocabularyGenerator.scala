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
import org.platanios.symphony.mt.utilities.{MutableFile, TrieWordCounter}

import better.files.File
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.io.BufferedWriter
import java.nio.charset.StandardCharsets
import java.nio.file.StandardOpenOption

import scala.collection.mutable
import scala.collection.parallel.mutable.ParMap
import scala.io.Source
import scala.util.matching.Regex

// TODO: Support shared BPE subword units across languages.

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
  * @param  numSymbols      Number of BPE symbols to generate. This is equivalent to the number of BPE merge operations
  *                         being learned when generating a new BPE vocabulary.
  * @param  separator       Separator symbol appended to all inter-word symbols while encoding sentences. This allows
  *                         for decoding BPE-encoded sentences, after a translation is done.
  * @param  countThreshold  Symbols pairs which appears less than `countThreshold` times will be ignored.
  * @param  replaceExisting If `true`, existing vocabulary files will be replaced, if found.
  * @param  bufferSize      Buffer size to use while reading and writing files.
  *
  * @author Emmanouil Antonios Platanios
  */
class BPEVocabularyGenerator protected (
    val numSymbols: Int = 32000,
    val separator: String = "@@",
    val countThreshold: Int = -1,
    val replaceExisting: Boolean = false,
    val bufferSize: Int = 8192
) extends VocabularyGenerator {
  protected val mergePairs        : mutable.Map[Language, Map[(String, String), Int]]    = mutable.Map.empty
  protected val reversedMergePairs: mutable.Map[Language, Map[String, (String, String)]] = mutable.Map.empty
  protected val vocabularies      : mutable.Map[Language, Set[String]]                   = mutable.Map.empty

  protected def mergePairsFilename(language: Language): String = s"merge_pairs.bpe.$numSymbols.${language.abbreviation}"

  /** Returns the vocabulary file name that this generator uses / will use.
    *
    * @param  language Language for which a vocabulary will be generated.
    * @return Vocabulary file name.
    */
  override def filename(language: Language): String = s"vocab.bpe.$numSymbols.${language.abbreviation}"

  /** Generates/Replaces a vocabulary file given a sequence of tokenized text files.
    *
    * '''NOTE:''' This method replaces the tokenized files with new files containing the BPE tokenized text.
    *
    * @param  language       Language for which a vocabulary will be generated.
    * @param  tokenizedFiles Tokenized text files to use for generating the vocabulary file.
    * @param  vocabDir       Directory in which to save the generated vocabulary file.
    * @return The generated/replaced vocabulary file.
    */
  override def generate(language: Language, tokenizedFiles: Seq[MutableFile], vocabDir: File): File = {
    if (!mergePairs.contains(language)) {
      mergePairs += language -> Map.empty
      reversedMergePairs += language -> Map.empty
      vocabularies += language -> Set.empty
    }

    // We first generate the merge pairs, if necessary.
    val mergePairsFile = vocabDir / mergePairsFilename(language)
    if (mergePairsFile.exists && !replaceExisting) {
      BPEVocabularyGenerator.logger.info(s"Loading existing BPE coding for $language: $mergePairsFile.")
      initializeMergePairs(language, vocabDir)
    } else {
      BPEVocabularyGenerator.logger.info(s"Learning BPE coding for $language.")
      mergePairsFile.parent.createDirectories()
      val mergePairsWriter = new BufferedWriter(
        mergePairsFile.newPrintWriter()(Seq(
          StandardOpenOption.CREATE,
          StandardOpenOption.WRITE,
          StandardOpenOption.TRUNCATE_EXISTING)), bufferSize)
      val tokens = mutable.ArrayBuffer(tokenizedFiles.map(_.get).toStream.flatMap(file => {
        Source.fromFile(file.toJava)(StandardCharsets.UTF_8)
            .getLines
            .flatMap(BPEVocabularyGenerator.whitespaceRegex.split)
      }).foldLeft(TrieWordCounter())((counter, word) => {
        counter.insertWord(word)
        counter
      }).words().map(p => (p._1, {
        val parts = p._2.split("")
        parts.update(parts.length - 1, parts.last + BPEVocabularyGenerator.END_OF_WORD_SYMBOL)
        parts.toSeq
      })).toSeq: _*)

      val pairStatistics = BPEVocabularyGenerator.computePairStatistics(tokens)
      val fullCounts = pairStatistics.counts
      val indices = pairStatistics.indices

      // Threshold is inspired by a Zipfian assumption, but it should only affect speed
      var threshold = fullCounts.values.max / 10
      var counts = ParMap(fullCounts.seq.toSeq: _*).withDefaultValue(0L)

      var continue = true
      var currentSymbol = 0
      var progressLogTime = System.currentTimeMillis

      while (currentSymbol < numSymbols && continue) {
        val time = System.currentTimeMillis
        if (time - progressLogTime >= 6e4) {
          val numBars = Math.floorDiv(10 * currentSymbol, numSymbols)
          BPEVocabularyGenerator.logger.info(
            s"│${"═" * numBars}${" " * (10 - numBars)}│ " +
                s"%${numSymbols.toString.length}s / $numSymbols BPE symbols processed.".format(currentSymbol))
          progressLogTime = time
        }
        var mostFrequent = if (counts.nonEmpty) counts.maxBy(_._2) else null
        if (counts.isEmpty || (currentSymbol > 0 && mostFrequent._2 < threshold)) {
          BPEVocabularyGenerator.pruneCounts(counts, fullCounts, threshold)
          counts = ParMap(fullCounts.seq.toSeq: _*).withDefaultValue(0L)
          mostFrequent = counts.maxBy(_._2)
          threshold = (mostFrequent._2 * currentSymbol / (currentSymbol + 10000f)).toLong
          counts = BPEVocabularyGenerator.pruneCounts(counts, fullCounts, threshold)
        }

        if (mostFrequent._2 < countThreshold) {
          BPEVocabularyGenerator.logger.info(
            s"No pair has frequency higher than $countThreshold. Stopping the byte pair encoding (BPE) iteration.")
          continue = false
        } else {
          if (!mergePairs(language).contains(mostFrequent._1)) {
            mergePairs(language) += mostFrequent._1 -> currentSymbol
            reversedMergePairs(language) += mostFrequent._1._1 + mostFrequent._1._2 -> mostFrequent._1
            mergePairsWriter.write(s"${mostFrequent._1._1}\t${mostFrequent._1._2}\n")
          }
          val changes = BPEVocabularyGenerator.replacePair(mostFrequent._1, tokens, indices)
          BPEVocabularyGenerator.updatePairStatistics(mostFrequent._1, changes, counts, indices)
          if (currentSymbol % 100 == 0)
            counts = BPEVocabularyGenerator.pruneCounts(counts, fullCounts, threshold)
        }

        currentSymbol += 1
      }

      mergePairsWriter.flush()
      mergePairsWriter.close()

      BPEVocabularyGenerator.logger.info(
        s"│${"═" * 10}│ %${numSymbols.toString.length}s / $numSymbols BPE symbols processed.".format(currentSymbol))
      BPEVocabularyGenerator.logger.info(s"Learned BPE coding for $language: $mergePairsFile.")
    }

    // We then generate the vocabulary, if necessary.
    val vocabFile = vocabDir / filename(language)
    val vocabWriter = {
      if (vocabFile.exists && !replaceExisting) {
        BPEVocabularyGenerator.logger.info(s"Vocabulary file for $language already exists: $vocabFile.")
        initializeVocabularies(language, vocabDir)
        None
      } else {
        BPEVocabularyGenerator.logger.info(s"Generating vocabulary file for $language.")
        vocabFile.parent.createDirectories()
        Some(new BufferedWriter(
          vocabFile.newPrintWriter()(Seq(
            StandardOpenOption.CREATE,
            StandardOpenOption.WRITE,
            StandardOpenOption.TRUNCATE_EXISTING)), bufferSize))
      }
    }

    // Irrespective of whether a new vocabulary is being generated, or an existing one was loaded, we also convert the
    // provided tokenized files to their encoded equivalent.
    var fileWriters = Seq.empty[BufferedWriter]
    val tokens = tokenizedFiles.toStream.flatMap(mutableFile => {
      val oldFile = mutableFile.get
      val file = oldFile.sibling(s"${oldFile.nameWithoutExtension}.bpe.$numSymbols.${language.abbreviation}")
      mutableFile.set(file)
      if (replaceExisting || file.notExists || vocabWriter.isDefined) {
        BPEVocabularyGenerator.logger.info(s"Applying BPE coding to file: $oldFile.")
        val fileWriter = new BufferedWriter(
          file.newPrintWriter()(Seq(
            StandardOpenOption.CREATE,
            StandardOpenOption.WRITE,
            StandardOpenOption.TRUNCATE_EXISTING)), bufferSize)
        val cache = mutable.Map.empty[String, Seq[String]]
        val tokens = Source.fromFile(oldFile.toJava)(StandardCharsets.UTF_8)
            .getLines
            .flatMap(line => {
              var sentence = BPEVocabularyGenerator.whitespaceRegex.split(line)
              sentence = encodeSentence(language, sentence, cache).toArray
              fileWriter.write(s"${sentence.mkString(" ")}\n")
              sentence
            })
        fileWriters :+= fileWriter
        tokens
      } else {
        Seq.empty
      }
    })

    vocabWriter.foreach(writer => {
      tokens.foldLeft(TrieWordCounter())((counter, word) => {
        counter.insertWord(word)
        counter
      }).words()
          .toSeq.sortBy(-_._1).map(_._2)
          .foreach(word => {
            vocabularies(language) += word
            writer.write(word + "\n")
          })
      writer.flush()
      writer.close()
      BPEVocabularyGenerator.logger.info(s"Generated vocabulary file for $language.")
    })

    fileWriters.foreach(fileWriter => {
      fileWriter.flush()
      fileWriter.close()
    })

    BPEVocabularyGenerator.logger.info(s"Applied BPE coding to all provided files for $language.")

    vocabFile
  }

  /** Initializes the merge pairs of this BPE generators from an existing file.
    *
    * @param  language Language for which a vocabulary has been generated.
    * @param  vocabDir Directory in which the generated vocabulary file and any other relevant files have been saved.
    */
  protected def initializeMergePairs(language: Language, vocabDir: File): Unit = {
    val mergePairsFile = vocabDir / mergePairsFilename(language)
    mergePairs += language -> Source.fromFile(mergePairsFile.toJava)(StandardCharsets.UTF_8)
        .getLines
        .filter(_ != "")
        .map(l => {
          val parts = l.split("\t")
          (parts(0), parts(1))
        }).zipWithIndex.toMap
    reversedMergePairs += language -> mergePairs(language).toSeq.map(p => p._1._1 + p._1._2 -> p._1).toMap
  }

  /** Initializes the vocabularies of this BPE generators from an existing file.
    *
    * @param  language Language for which a vocabulary has been generated.
    * @param  vocabDir Directory in which the generated vocabulary file and any other relevant files have been saved.
    */
  protected def initializeVocabularies(language: Language, vocabDir: File): Unit = {
    val vocabFile = vocabDir / filename(language)
    vocabularies += language -> Source.fromFile(vocabFile.toJava)(StandardCharsets.UTF_8)
        .getLines
        .filter(_ != "")
        .toSet
  }

  /** Returns a vocabulary for the specified language, ready to be used by machine translation models.
    *
    * @param  language Language for which to return a vocabulary.
    * @param  vocabDir Directory in which the generated vocabulary file and any other relevant files have been saved.
    * @return Created vocabulary.
    */
  override def getVocabulary(language: Language, vocabDir: File): Vocabulary = {
    CodedVocabulary(vocabDir / filename(language), s => encodeSentence(language, s), decodeSentence)
  }

  /** Encodes the provided sentence to a sequence of BPE coded words.
    *
    * @param  language Language in which the sentence is written.
    * @param  sentence Sentence to encode as a sequence of words.
    * @param  cache    Optional cache of already encoded words, used to speed up the encoding process.
    * @return Encoded sentence as a sequence of BPE coded words.
    */
  def encodeSentence(
      language: Language,
      sentence: Seq[String],
      cache: mutable.Map[String, Seq[String]] = mutable.Map.empty
  ): Seq[String] = {
    // TODO: Add support for glossaries (i.e., words that will be encoded with the identity function.
    sentence.flatMap(word => {
      var parts = encodeWord(language, word, cache)
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
    * @param  language Language in which the word is written.
    * @param  word     Word to encode.
    * @param  cache    Optional cache of already encoded words, used to speed up the encoding process.
    * @return Encoded word as a sequence of BPE coded words.
    */
  def encodeWord(
      language: Language,
      word: String,
      cache: mutable.Map[String, Seq[String]] = mutable.Map.empty
  ): Seq[String] = {
    cache.getOrElseUpdate(word, {
      var wordParts = word.split("").toSeq
      wordParts = wordParts.updated(
        wordParts.length - 1, wordParts.last + BPEVocabularyGenerator.END_OF_WORD_SYMBOL)
      var pairs = wordParts.sliding(2).map(p => (p(0), p(1)))
      if (pairs.isEmpty) {
        wordParts = wordParts.updated(
          wordParts.length - 1, wordParts.last.dropRight(BPEVocabularyGenerator.END_OF_WORD_SYMBOL.length))
        wordParts
      } else {
        var continue = true
        while (pairs.nonEmpty && continue) {
          val pair = pairs.map(p => (p, mergePairs(language).get(p))).minBy(_._2.getOrElse(Int.MaxValue))
          if (pair._2.isEmpty) {
            continue = false
          } else {
            val newWordParts = mutable.ListBuffer.empty[String]
            var i = 0
            var last = 0
            while (i < wordParts.length) {
              if (wordParts(i) != pair._1._1 && i == wordParts.length - 1) {
                var k = last
                while (k < wordParts.length) {
                  newWordParts += wordParts(k)
                  k += 1
                }
                // Force the loop to finish.
                i = wordParts.length
              } else if (wordParts(i) != pair._1._1) {
                i += 1
              } else {
                var k = last
                while (k < i) {
                  newWordParts += wordParts(k)
                  k += 1
                }
                if (wordParts(i) == pair._1._1 && i < wordParts.length - 1 && wordParts(i + 1) == pair._1._2) {
                  newWordParts += pair._1._1 + pair._1._2
                  i += 2
                } else {
                  newWordParts += wordParts(i)
                  i += 1
                }
                last = i
              }
            }
            wordParts = newWordParts
            pairs = wordParts.sliding(2).map(p => (p(0), p(1)))
          }
        }

        // Remove end-of-word symbols.
        if (wordParts.last == BPEVocabularyGenerator.END_OF_WORD_SYMBOL)
          wordParts = wordParts.slice(0, wordParts.length - 1)
        else if (wordParts.last.endsWith(BPEVocabularyGenerator.END_OF_WORD_SYMBOL))
          wordParts = wordParts.updated(wordParts.length - 1, wordParts.last
              .dropRight(BPEVocabularyGenerator.END_OF_WORD_SYMBOL.length))

        // Check if the new words parts are in the vocabulary, and backtrack if necessary.
        wordParts = BPEVocabularyGenerator.checkVocabularyAndSplit(
          wordParts, reversedMergePairs(language), vocabularies(language), separator)

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
}

object BPEVocabularyGenerator {
  private[BPEVocabularyGenerator] val logger = Logger(LoggerFactory.getLogger("Vocabulary / BPE Generator"))

  def apply(
      numSymbols: Int = 32000,
      separator: String = "@@",
      countThreshold: Int = -1,
      replaceExisting: Boolean = false,
      bufferSize: Int = 8192
  ): BPEVocabularyGenerator = {
    new BPEVocabularyGenerator(numSymbols, separator, countThreshold, replaceExisting, bufferSize)
  }

  /** End-of-word symbol used by the BPE vocabulary generator. */
  val END_OF_WORD_SYMBOL: String = "</w>"

  /** Regular expressions used for adding spaces between the characters of words and for tokenizing sentences. */
  private[BPEVocabularyGenerator] val emptyRegex     : Regex = "(?!$)".r
  private[BPEVocabularyGenerator] val whitespaceRegex: Regex = "\\s+".r

  private[BPEVocabularyGenerator] case class PairStatistics(
      counts: ParMap[(String, String), Long],
      indices: ParMap[(String, String), ParMap[Int, Long]])

  private[BPEVocabularyGenerator] case class Change(
      index: Int,
      word: Seq[String],
      newWord: Seq[String],
      count: Long)

  /** Computes the pair statistics for the provided vocabulary of words.
    *
    * @param  words Vocabulary of words for which to compute statistics (each tuple contains a count and a word).
    * @return Computed statistics.
    */
  private[BPEVocabularyGenerator] def computePairStatistics(
      words: Seq[(Long, Seq[String])]
  ): PairStatistics = {
    val counts = ParMap.empty[(String, String), Long].withDefaultValue(0)
    val indices = ParMap.empty[(String, String), ParMap[Int, Long]]
        .withDefaultValue(ParMap.empty[Int, Long].withDefaultValue(0))
    words.zipWithIndex.foreach {
      case ((count, symbols), index) =>
        var prevSymbol = symbols(0)
        symbols.tail.foreach(symbol => {
          val pair = (prevSymbol, symbol)
          counts += pair -> (counts(pair) + count)
          indices(pair) += index -> (indices(pair)(index) + 1)
          prevSymbol = symbol
        })
    }
    PairStatistics(counts, indices)
  }

  /** Prunes the pair counts map for improved efficiency.
    *
    * The frequency of a symbol pair never increases, so pruning is generally safe (until the most frequent pair is less
    * frequent than a pair that was previously pruned).
    *
    * @param  counts     Symbol counts map to be pruned.
    * @param  fullCounts Map that keeps counts information for when we need to access pruned symbol counts. Note that
    *                    this map may be modified by this method.
    * @param  threshold  Symbol count threshold to use while pruning.
    * @return Pruned symbol counts map.
    */
  private[BPEVocabularyGenerator] def pruneCounts(
      counts: ParMap[(String, String), Long],
      fullCounts: ParMap[(String, String), Long],
      threshold: Long
  ): ParMap[(String, String), Long] = {
    counts.filter {
      case (pair, count) =>
        val keep = count >= threshold
        if (!keep) {
          if (count < 0)
            fullCounts += pair -> (fullCounts(pair) + count)
          else
            fullCounts += pair -> count
        }
        keep
    }.withDefaultValue(0L)
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
      indices: ParMap[(String, String), ParMap[Int, Long]]
  ): Seq[Change] = {
    val joinedPair = pair._1 + pair._2
    indices(pair).toSeq.filter(_._2 >= 1).map(_._1).map(index => {
      val (count, word) = words(index)
      val newWord = mutable.ListBuffer.empty[String]
      var j = 0
      while (j < word.length - 1) {
        (word(j), word(j + 1)) match {
          case p if p == pair => newWord += joinedPair; j += 2
          case _ => newWord += word(j); j += 1
        }
      }
      words.update(index, (count, newWord))
      Change(index, word, newWord, count)
    }).seq
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
      counts: ParMap[(String, String), Long],
      indices: ParMap[(String, String), ParMap[Int, Long]]
  ): Unit = {
    val joinedPair = pair._1 + pair._2

    counts += pair -> 0L
    indices += pair -> ParMap.empty[Int, Long].withDefaultValue(0L)
    changes.foreach(change => {
      // Find all instances of the pair, and update the corresponding statistics.
      var i = 0
      while (i < change.word.length - 1) {
        (change.word(i), change.word(i + 1)) match {
          case p if p == pair =>
            // Assuming a symbol sequence "A B C", if "B C" is merged, we reduce the frequency of "A B".
            if (i > 0) {
              val prevPair = (change.word(i - 1), change.word(i))
              counts += prevPair -> (counts(prevPair) - change.count)
              indices(prevPair) += change.index -> (indices(prevPair)(change.index) - 1)
            }
            // Assuming a symbol sequence "A B C B", if "B C" is merged, we reduce the frequency of "C B". However, we
            // skip this if the sequence is "A B C B C", because the frequency of "C B" will have already been reduced
            // by the previous code block.
            if (i < change.word.length - 2 &&
                (change.word(i + 2) != pair._1 || i >= change.word.length - 3 || change.word(i + 3) != pair._2)) {
              val nextPair = (change.word(i + 1), change.word(i + 2))
              counts += nextPair -> (counts(nextPair) - change.count)
              indices(nextPair) += change.index -> (indices(nextPair)(change.index) - 1)
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
              counts += prevPair -> (counts(prevPair) + change.count)
              indices(prevPair) += change.index -> (indices(prevPair)(change.index) + 1)
            }
            // Assuming a symbol sequence "A BC B", if "B C" is merged, we increase the frequency of "BC B". However, we
            // skip this if the sequence is "A BC BC", because the count of "BC BC" will have already been incremented
            // by the previous code block.
            if (i < change.newWord.length - 1 && change.newWord(i + 1) != joinedPair) {
              val nextPair = (change.newWord(i), change.newWord(i + 1))
              counts += nextPair -> (counts(nextPair) + change.count)
              indices(nextPair) += change.index -> (indices(nextPair)(change.index) + 1)
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
      if (isLast)
        reversedMergePairs.get(wordPart + END_OF_WORD_SYMBOL).map(p => (p._1, p._2.drop(END_OF_WORD_SYMBOL.length)))
      else
        reversedMergePairs.get(wordPart)
    }

    pair match {
      case None => Seq(wordPart)
      case Some((left, right)) => { // We first go through the left parts.
        if (vocabulary.contains(left + separator))
          Seq(left)
        else
          splitRecursively(left, reversedMergePairs, vocabulary, separator, isLast = false)
      } ++ { // We then go through the right parts.
        if ((isLast && vocabulary.contains(right)) || (!isLast && vocabulary.contains(right + separator)))
          Seq(right)
        else
          splitRecursively(right, reversedMergePairs, vocabulary, separator, isLast = isLast)
      }
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
        case (part, index) if index < wordParts.length - 1 =>
          if (vocabulary.contains(part + separator))
            Seq(part)
          else
            splitRecursively(part, reversedMergePairs, vocabulary, separator, isLast = false)
        case (part, _) =>
          if (vocabulary.contains(part))
            Seq(part)
          else
            splitRecursively(part, reversedMergePairs, vocabulary, separator, isLast = true)
      }
    }
  }
}
