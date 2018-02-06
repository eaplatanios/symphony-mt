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

import better.files._
import org.eclipse.jgit.api.Git

import scala.sys.process._

/**
  * @author Emmanouil Antonios Platanios
  */
object Utilities {
  case class MosesDecoder(path: File) {
    val gitUrl: String = "https://github.com/moses-smt/mosesdecoder.git"

    def exists: Boolean = path.exists()

    def cloneRepository(): Unit = {
      Git.cloneRepository()
          .setURI(gitUrl)
          .setDirectory(path.toJava)
          .call()
    }

    def sgmToText(sgmFile: File, textFile: File): Int = {
      ((path / "scripts" / "ems" / "support" / "input-from-sgm.perl").toString #< sgmFile.toJava #> textFile.toJava).!
    }

    def tokenize(textFile: File, tokenizedFile: File, languageAbbreviation: String, numThreads: Int = 8): Int = {
      val tokenizer = (path / "scripts" / "tokenizer" / "tokenizer.perl").toString
      (Seq(tokenizer, "-q", "-l", languageAbbreviation, "-threads", numThreads.toString) #<
          textFile.toJava #>
          tokenizedFile.toJava).!
    }

    def cleanCorpus(
        corpusFile: File,
        cleanCorpusFile: File,
        srcLanguageAbbreviation: String,
        tgtLanguageAbbreviation: String,
        minLength: Int,
        maxLength: Int
    ): Int = {
      Seq(
        (path / "scripts" / "training" / "clean-corpus-n.perl").toString, corpusFile.toString(),
        srcLanguageAbbreviation, tgtLanguageAbbreviation,
        cleanCorpusFile.toString(), minLength.toString, maxLength.toString).!
    }
  }
}
