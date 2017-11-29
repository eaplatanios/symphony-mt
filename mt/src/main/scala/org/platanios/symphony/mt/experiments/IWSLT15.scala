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

package org.platanios.symphony.mt.experiments

import org.platanios.symphony.mt.core.{Configuration, Language, Translator}
import org.platanios.symphony.mt.data.Datasets.MTTextLinesDataset
import org.platanios.symphony.mt.data.Vocabulary
import org.platanios.symphony.mt.data.loaders.IWSLT15Loader
import org.platanios.symphony.mt.translators.PairwiseRNNTranslator
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.layers.rnn.cell.{BasicLSTMCell, MultiRNNCell}
import org.platanios.tensorflow.api.ops.io.data.TextLinesDataset

import java.nio.file.{Path, Paths}

/**
  * @author Emmanouil Antonios Platanios
  */
object IWSLT15 extends App {
  val workingDir: Path = Paths.get("temp")
  val dataDir   : Path = workingDir.resolve("data").resolve("iwslt15.en-vi")

  IWSLT15Loader.load(dataDir, IWSLT15Loader.EnglishVietnamese)

  // Create the languages and their corresponding vocabularies
  val enCheck: Option[(Int, Path)] = Vocabulary.check(dataDir.resolve("vocab.en"))
  val viCheck: Option[(Int, Path)] = Vocabulary.check(dataDir.resolve("vocab.vi"))
  if (enCheck.isEmpty)
    throw new IllegalArgumentException("Could not load the english vocabulary file.")
  if (viCheck.isEmpty)
    throw new IllegalArgumentException("Could not load the vietnamese vocabulary file.")
  val enVocabSize: Int      = enCheck.get._1
  val enVocabPath: Path     = enCheck.get._2
  val viVocabSize: Int      = viCheck.get._1
  val viVocabPath: Path     = viCheck.get._2
  val srcLang    : Language = Language("English", "en", () => Vocabulary.createTable(enVocabPath), enVocabSize)
  val tgtLang    : Language = Language("Vietnamese", "vi", () => Vocabulary.createTable(viVocabPath), viVocabSize)

  // Create the datasets
  val srcTrainDataset: MTTextLinesDataset = TextLinesDataset(dataDir.resolve("train.en").toAbsolutePath.toString)
  val tgtTrainDataset: MTTextLinesDataset = TextLinesDataset(dataDir.resolve("train.vi").toAbsolutePath.toString)
  val srcDevDataset  : MTTextLinesDataset = TextLinesDataset(dataDir.resolve("tst2012.en").toAbsolutePath.toString)
  val tgtDevDataset  : MTTextLinesDataset = TextLinesDataset(dataDir.resolve("tst2012.vi").toAbsolutePath.toString)
  val srcTestDataset : MTTextLinesDataset = TextLinesDataset(dataDir.resolve("tst2013.en").toAbsolutePath.toString)
  val tgtTestDataset : MTTextLinesDataset = TextLinesDataset(dataDir.resolve("tst2013.vi").toAbsolutePath.toString)

  // Create a translator
  val configuration: Configuration = Configuration()
  val translator   : Translator    = new PairwiseRNNTranslator(
    encoderCell = MultiRNNCell(Seq(
      BasicLSTMCell(
        configuration.modelNumUnits, FLOAT32, Shape(configuration.modelSrcEmbeddingSize), forgetBias = 1.0f,
        name = "EncoderCell1"),
      BasicLSTMCell(
        configuration.modelNumUnits, FLOAT32, Shape(configuration.modelNumUnits), forgetBias = 1.0f,
        name = "EncoderCell2"))),
    decoderCell = MultiRNNCell(Seq(
      BasicLSTMCell(
        configuration.modelNumUnits, FLOAT32, Shape(4 * configuration.modelNumUnits), forgetBias = 1.0f,
        name = "DecoderCell1"),
      BasicLSTMCell(
        configuration.modelNumUnits, FLOAT32, Shape(configuration.modelNumUnits), forgetBias = 1.0f,
        name = "DecoderCell2"))),
    configuration = configuration)

  translator.train(
    Seq(Translator.DatasetPair(srcLang, tgtLang, srcTrainDataset, tgtTrainDataset)),
    Seq(Translator.DatasetPair(srcLang, tgtLang, srcDevDataset, tgtDevDataset)),
    Seq(Translator.DatasetPair(srcLang, tgtLang, srcTestDataset, tgtTestDataset)),
    tf.learn.StopCriteria(Some(10000)))
}
