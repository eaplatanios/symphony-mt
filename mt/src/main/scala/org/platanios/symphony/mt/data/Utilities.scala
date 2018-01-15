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

package org.platanios.symphony.mt.data

import org.eclipse.jgit.api.Git
import org.platanios.symphony.mt.Language

import java.io.{BufferedWriter, File, PrintWriter}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

import scala.io.Source
import scala.sys.process._

/**
  * @author Emmanouil Antonios Platanios
  */
object Utilities {
  def createVocab(tokenizedFiles: Set[Path], vocabFile: Path, bufferSize: Int = 8192): Unit = {
    val writer = new BufferedWriter(new PrintWriter(vocabFile.toFile), bufferSize)
    tokenizedFiles.flatMap(file => {
      Source.fromFile(file.toFile)(StandardCharsets.UTF_8)
          .getLines
          .flatMap(_.split("\\s+"))
    }).foldLeft(Map.empty[String, Int])((count, word) => count + (word -> (count.getOrElse(word, 0) + 1)))
        .toSeq.sortWith(_._2 > _._2)
        .foreach(wordPair => writer.write(wordPair._1 + "\n"))
    writer.flush()
    writer.close()
  }

  case class MosesDecoder(path: Path) {
    val gitUrl: String = "https://github.com/moses-smt/mosesdecoder.git"

    def exists: Boolean = Files.exists(path)

    def cloneRepository(): Unit = {
      Git.cloneRepository()
          .setURI(gitUrl)
          .setDirectory(new File(path.toAbsolutePath.toString))
          .call()
    }

    def sgmToText(sgmFile: Path, textFile: Path): Unit = {
      (path
          .resolve("scripts")
          .resolve("ems")
          .resolve("support")
          .resolve("input-from-sgm.perl")
          .toAbsolutePath.toString #< sgmFile.toFile #> textFile.toFile).!
    }

    def tokenize(textFile: Path, vocabFile: Path, language: Language, numThreads: Int = 8): Unit = {
      val tokenizer = path.resolve("scripts").resolve("tokenizer").resolve("tokenizer.perl").toAbsolutePath.toString
      (Seq(tokenizer, "-q", "-l", language.abbreviation, "-threads", numThreads.toString) #<
          textFile.toFile #>
          vocabFile.toFile).!
    }

    def cleanCorpus(
        corpus: String,
        cleanCorpus: String,
        srcLanguage: Language,
        tgtLanguage: Language,
        minLength: Int,
        maxLength: Int
    ): Unit = {
      val corpusCleaner = path
          .resolve("scripts")
          .resolve("training")
          .resolve("clean-corpus-n.perl")
          .toAbsolutePath.toString
      Seq(
        corpusCleaner, corpus, srcLanguage.abbreviation, tgtLanguage.abbreviation,
        cleanCorpus, minLength.toString, maxLength.toString).!
    }
  }
}
