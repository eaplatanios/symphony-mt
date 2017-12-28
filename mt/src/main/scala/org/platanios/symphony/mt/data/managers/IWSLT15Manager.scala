/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.symphony.mt.data.managers

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.Language._
import org.platanios.symphony.mt.data.ParallelDataset

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.{Files, Path}

/**
  * @author Emmanouil Antonios Platanios
  */
case class IWSLT15Manager(srcLanguage: Language, tgtLanguage: Language) {
  val src: String = srcLanguage.abbreviation
  val tgt: String = tgtLanguage.abbreviation

  val name: String = s"$src-$tgt"

  private[this] val reversed: Boolean = EuroparlV7Manager.supportedLanguagePairs.contains((tgtLanguage, srcLanguage))

  private[this] val directoryName: String = if (reversed) s"iwslt15.$tgt-$src" else s"iwslt15.$src-$tgt"

  def download(path: Path, bufferSize: Int = 8192): ParallelDataset = {
    val processedPath = path.resolve("iwslt-15")

    // Download the data, if necessary.
    val srcTrainCorpus = processedPath.resolve(directoryName).resolve(s"${IWSLT15Manager.trainPrefix}.$src")
    val tgtTrainCorpus = processedPath.resolve(directoryName).resolve(s"${IWSLT15Manager.trainPrefix}.$tgt")
    val srcDevCorpus = processedPath.resolve(directoryName).resolve(s"${IWSLT15Manager.devPrefix}.$src")
    val tgtDevCorpus = processedPath.resolve(directoryName).resolve(s"${IWSLT15Manager.devPrefix}.$tgt")
    val srcTestCorpus = processedPath.resolve(directoryName).resolve(s"${IWSLT15Manager.testPrefix}.$src")
    val tgtTestCorpus = processedPath.resolve(directoryName).resolve(s"${IWSLT15Manager.testPrefix}.$tgt")
    val srcVocab = processedPath.resolve(directoryName).resolve(s"${IWSLT15Manager.vocabPrefix}.$src")
    val tgtVocab = processedPath.resolve(directoryName).resolve(s"${IWSLT15Manager.vocabPrefix}.$tgt")

    if (!Files.exists(srcTrainCorpus))
      Manager.maybeDownload(
        srcTrainCorpus, s"${IWSLT15Manager.url}/$directoryName/${IWSLT15Manager.trainPrefix}.$src", bufferSize)
    if (!Files.exists(tgtTrainCorpus))
      Manager.maybeDownload(
        tgtTrainCorpus, s"${IWSLT15Manager.url}/$directoryName/${IWSLT15Manager.trainPrefix}.$tgt", bufferSize)
    if (!Files.exists(srcDevCorpus))
      Manager.maybeDownload(
        srcDevCorpus, s"${IWSLT15Manager.url}/$directoryName/${IWSLT15Manager.devPrefix}.$src", bufferSize)
    if (!Files.exists(tgtDevCorpus))
      Manager.maybeDownload(
        tgtDevCorpus, s"${IWSLT15Manager.url}/$directoryName/${IWSLT15Manager.devPrefix}.$tgt", bufferSize)
    if (!Files.exists(srcTestCorpus))
      Manager.maybeDownload(
        srcTestCorpus, s"${IWSLT15Manager.url}/$directoryName/${IWSLT15Manager.testPrefix}.$src", bufferSize)
    if (!Files.exists(tgtTestCorpus))
      Manager.maybeDownload(
        tgtTestCorpus, s"${IWSLT15Manager.url}/$directoryName/${IWSLT15Manager.testPrefix}.$tgt", bufferSize)
    if (!Files.exists(srcVocab))
      Manager.maybeDownload(
        srcVocab, s"${IWSLT15Manager.url}/$directoryName/${IWSLT15Manager.vocabPrefix}.$src", bufferSize)
    if (!Files.exists(tgtVocab))
      Manager.maybeDownload(
        tgtVocab, s"${IWSLT15Manager.url}/$directoryName/${IWSLT15Manager.vocabPrefix}.$tgt", bufferSize)

    ParallelDataset(Seq(srcLanguage, tgtLanguage))(
      trainCorpora = Seq(Seq(srcTrainCorpus), Seq(tgtTrainCorpus)),
      devCorpora = Seq(Seq(srcDevCorpus), Seq(tgtDevCorpus)),
      testCorpora = Seq(Seq(srcTestCorpus), Seq(tgtTestCorpus)),
      vocabularies = Seq(Seq(srcVocab), Seq(tgtVocab)))
  }
}

object IWSLT15Manager {
  private[IWSLT15Manager] val logger = Logger(LoggerFactory.getLogger("IWSLT-15 Data Manager"))

  val url        : String = "https://nlp.stanford.edu/projects/nmt/data"
  val trainPrefix: String = "train"
  val devPrefix  : String = "tst2012"
  val testPrefix : String = "tst2013"
  val vocabPrefix: String = "vocab"

  val supportedLanguagePairs: Set[(Language, Language)] = Set((Vietnamese, English))
}
