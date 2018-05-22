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

package org.platanios.symphony.mt.experiments

import org.platanios.symphony.mt.Language
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.io.CheckpointReader

import better.files.File
import net.schmizz.sshj.SSHClient
import net.schmizz.sshj.xfer.FileSystemFile

import java.io.PrintWriter

/**
  * @author Emmanouil Antonios Platanios
  */
object ParameterExtractor {
  def checkpointFiles(prefix: String): Set[String] = {
    Set(
      s"$prefix.data-00000-of-00002",
      s"$prefix.data-00001-of-00002",
      s"$prefix.index",
      s"$prefix.meta")
  }

  def main(args: Array[String]): Unit = {
    val server = GPU3Server
    val workingDir = File.currentWorkingDirectory / "temp" / "results"
    workingDir.createIfNotExists(asDirectory = true)
//    val remotePath = "~/code/symphony-mt/temp/experiments/ted_talks.en.es.fr.it.nl.ro.de.vi.hi.ta.tw:true.ae:true.bi_rnn:2:2.lstm:tanh.hyper_lang.l:8.w:512.r.a.d:0.2.ls:0.1.t:none.c:none.v:bpe-10000.pp:100.bs:128.nb:5.sml:50.tml:50/"
    val remotePath = "~/code/symphony-mt/temp/experiments/iwslt17.de-en.tw:false.ae:false.bi_rnn:2:2.lstm:tanh.pairwise.w:512.r.a.d:0.2.ls:0.1.t:moses.c:moses.v:simple-20000-5.pp:100.bs:128.nb:5.sml:50.tml:50/"
    val filePrefix = "model.ckpt-25000"
    val files = checkpointFiles(filePrefix) // + "languages.index"
    val ssh = new SSHClient()
    ssh.useCompression()
    ssh.loadKnownHosts()
    ssh.connect(server.hostname)
    try {
      ssh.authPublickey(server.username)
      files.foreach(f => ssh.newSCPFileTransfer().download(remotePath + f, new FileSystemFile((workingDir / f).toJava)))

      val checkpointReader = CheckpointReader((workingDir / filePrefix).path)
      val numParameters = checkpointReader.variableShapes
          .filter(!_._1.startsWith("AMSGrad"))
          .values
          .map(_.numElements)
          .sum
       checkpointReader.close()

//      // Obtain the languages index.
//      val languages = (workingDir / "languages.index").lines.map(line => {
//        val lineParts = line.split(',')
//        (lineParts(1).toInt, lineParts(0))
//      }).toSeq.sortBy(_._1).map(_._2)
//
//      // Obtain the language embeddings from the checkpoint file.
//      val checkpointReader = CheckpointReader((workingDir / filePrefix).path)
//      val variableShapes = checkpointReader.variableShapes.filter(_._1.endsWith("LanguageEmbeddings"))
//      val variableName = variableShapes.find(_._2 == Shape(languages.size, 8)).get._1
//      val variableValue = checkpointReader.getTensor(variableName).get
//      val languageEmbeddings = languages.zipWithIndex.map {
//        case (language, index) => language -> variableValue(index)
//      }
//
//      // Write the language embeddings to a CSV file to use for plotting.
//      val writer = new PrintWriter((workingDir / "iwslt15_language_embeddings.csv").toJava)
//      languageEmbeddings.foreach(l => {
//        writer.write(s"${Language.fromName(l._1).abbreviation},${l._2.entriesIterator.map(_.toString).mkString(",")}\n")
//      })
//      checkpointReader.close()
//      writer.close()
    } finally {
      ssh.disconnect()
    }
  }
}
