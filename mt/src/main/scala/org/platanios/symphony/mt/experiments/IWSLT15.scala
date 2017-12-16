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

import org.platanios.symphony.mt.{Environment, Language, LogConfig}
import org.platanios.symphony.mt.data.Datasets.MTTextLinesDataset
import org.platanios.symphony.mt.data.{DataConfig, Vocabulary}
import org.platanios.symphony.mt.data.loaders.IWSLT15Loader
import org.platanios.symphony.mt.models.{InferConfig, TrainConfig}
import org.platanios.symphony.mt.models.rnn.{BasicModel, LSTM}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.io.data.TextLinesDataset

import java.nio.file.{Path, Paths}

/**
  * @author Emmanouil Antonios Platanios
  */
object IWSLT15 extends App {
  val workingDir: Path = Paths.get("temp")
  val dataDir   : Path = workingDir.resolve("data").resolve("iwslt15.en-vi")

  IWSLT15Loader.download(workingDir.resolve("data"), IWSLT15Loader.EnglishVietnamese)

  // Create the languages and their corresponding vocabularies
  val srcLang : Language   = Language("Vietnamese", "vi")
  val tgtLang : Language   = Language("English", "en")
  val srcVocab: Vocabulary = Vocabulary(dataDir.resolve("vocab.vi"))
  val tgtVocab: Vocabulary = Vocabulary(dataDir.resolve("vocab.en"))

  // Create the datasets
  val srcTrainDataset: MTTextLinesDataset = TextLinesDataset(dataDir.resolve("train.vi").toAbsolutePath.toString)
  val tgtTrainDataset: MTTextLinesDataset = TextLinesDataset(dataDir.resolve("train.en").toAbsolutePath.toString)
  val srcDevDataset  : MTTextLinesDataset = TextLinesDataset(dataDir.resolve("tst2012.vi").toAbsolutePath.toString)
  val tgtDevDataset  : MTTextLinesDataset = TextLinesDataset(dataDir.resolve("tst2012.en").toAbsolutePath.toString)
  val srcTestDataset : MTTextLinesDataset = TextLinesDataset(dataDir.resolve("tst2013.vi").toAbsolutePath.toString)
  val tgtTestDataset : MTTextLinesDataset = TextLinesDataset(dataDir.resolve("tst2013.en").toAbsolutePath.toString)

  // Create a translator
  val config = BasicModel.Config(cell = LSTM(), numUnits = 128)

  val env = Environment(workingDir = Paths.get("temp").resolve(s"${srcLang.abbreviation}-${tgtLang.abbreviation}"))
  val dataConfig = DataConfig()
  val trainConfig = TrainConfig()
  val inferConfig = InferConfig()
  val logConfig = LogConfig()

  val model = BasicModel(
    config, srcLang, tgtLang, srcVocab, tgtVocab,
    srcTrainDataset, tgtTrainDataset, srcDevDataset, tgtDevDataset, srcTestDataset, tgtTestDataset,
    env, dataConfig, trainConfig, inferConfig, logConfig, "BasicModel")

  model.train(tf.learn.StopCriteria(Some(10000)))
}
