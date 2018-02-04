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

import org.platanios.symphony.mt.Language

import better.files._
import org.eclipse.jgit.api.Git

import java.io.BufferedWriter
import java.nio.charset.StandardCharsets

import scala.io.Source
import scala.sys.process._

/**
  * @author Emmanouil Antonios Platanios
  */
object Utilities {
  def createVocab(tokenizedFiles: Set[File], vocabFile: File, bufferSize: Int = 8192): Unit = {
    val writer = new BufferedWriter(vocabFile.newPrintWriter(), bufferSize)
    tokenizedFiles.flatMap(file => {
      Source.fromFile(file.toJava)(StandardCharsets.UTF_8)
          .getLines
          .flatMap(_.split("\\s+"))
    }).foldLeft(Map.empty[String, Int])((count, word) => count + (word -> (count.getOrElse(word, 0) + 1)))
        .toSeq.sortWith(_._2 > _._2)
        .foreach(wordPair => writer.write(wordPair._1 + "\n"))
    writer.flush()
    writer.close()
  }

  case class MosesDecoder(path: File) {
    val gitUrl: String = "https://github.com/moses-smt/mosesdecoder.git"

    def exists: Boolean = path.exists()

    def cloneRepository(): Unit = {
      Git.cloneRepository()
          .setURI(gitUrl)
          .setDirectory(path.toJava)
          .call()
    }

    def sgmToText(sgmFile: File, textFile: File): Unit = {
      ((path / "scripts" / "ems" / "support" / "input-from-sgm.perl").toString #< sgmFile.toJava #> textFile.toJava).!
    }

    def tokenize(textFile: File, vocabFile: File, language: Language, numThreads: Int = 8): Unit = {
      val tokenizer = (path / "scripts" / "tokenizer" / "tokenizer.perl").toString
      (Seq(tokenizer, "-q", "-l", language.abbreviation, "-threads", numThreads.toString) #<
          textFile.toJava #>
          vocabFile.toJava).!
    }

    def cleanCorpus(
        corpus: String,
        cleanCorpus: String,
        srcLanguage: Language,
        tgtLanguage: Language,
        minLength: Int,
        maxLength: Int
    ): Unit = {
      Seq(
        (path / "scripts" / "training" / "clean-corpus-n.perl").toString, corpus,
        srcLanguage.abbreviation, tgtLanguage.abbreviation, cleanCorpus, minLength.toString, maxLength.toString).!
    }
  }
}
