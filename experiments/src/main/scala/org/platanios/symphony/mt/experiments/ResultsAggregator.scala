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

import org.platanios.symphony.mt.experiments.results.ExperimentResult

import better.files._
import net.schmizz.sshj.SSHClient
import net.schmizz.sshj.xfer.FileSystemFile

// TODO: Make this more generic.

/**
  * @author Emmanouil Antonios Platanios
  */
object ResultsAggregator {
  def iwslt15ExperimentDirectories(method: String, percentParallel: Int): Seq[String] = method match {
    case "pairwise" => Seq(
      s"iwslt15.en-cs.both:false.back:false.bi_rnn:2:2.lstm:tanh.pairwise.w:512.r.a.dropout:0.2.ls:0.1.t:moses.c:moses.v:generated-simple-20000-5.pp:$percentParallel.bs:128.nb:5.sml:80.tml:80",
      s"iwslt15.cs-en.both:false.back:false.bi_rnn:2:2.lstm:tanh.pairwise.w:512.r.a.dropout:0.2.ls:0.1.t:moses.c:moses.v:generated-simple-20000-5.pp:$percentParallel.bs:128.nb:5.sml:80.tml:80",
      s"iwslt15.en-de.both:false.back:false.bi_rnn:2:2.lstm:tanh.pairwise.w:512.r.a.dropout:0.2.ls:0.1.t:moses.c:moses.v:generated-simple-20000-5.pp:$percentParallel.bs:128.nb:5.sml:80.tml:80",
      s"iwslt15.de-en.both:false.back:false.bi_rnn:2:2.lstm:tanh.pairwise.w:512.r.a.dropout:0.2.ls:0.1.t:moses.c:moses.v:generated-simple-20000-5.pp:$percentParallel.bs:128.nb:5.sml:80.tml:80",
      s"iwslt15.en-fr.both:false.back:false.bi_rnn:2:2.lstm:tanh.pairwise.w:512.r.a.dropout:0.2.ls:0.1.t:moses.c:moses.v:generated-simple-20000-5.pp:$percentParallel.bs:128.nb:5.sml:80.tml:80",
      s"iwslt15.fr-en.both:false.back:false.bi_rnn:2:2.lstm:tanh.pairwise.w:512.r.a.dropout:0.2.ls:0.1.t:moses.c:moses.v:generated-simple-20000-5.pp:$percentParallel.bs:128.nb:5.sml:80.tml:80",
      s"iwslt15.en-th.both:false.back:false.bi_rnn:2:2.lstm:tanh.pairwise.w:512.r.a.dropout:0.2.ls:0.1.t:moses.c:moses.v:generated-simple-20000-5.pp:$percentParallel.bs:128.nb:5.sml:80.tml:80",
      s"iwslt15.th-en.both:false.back:false.bi_rnn:2:2.lstm:tanh.pairwise.w:512.r.a.dropout:0.2.ls:0.1.t:moses.c:moses.v:generated-simple-20000-5.pp:$percentParallel.bs:128.nb:5.sml:80.tml:80",
      s"iwslt15.en-vi.both:false.back:false.bi_rnn:2:2.lstm:tanh.pairwise.w:512.r.a.dropout:0.2.ls:0.1.t:moses.c:moses.v:generated-simple-20000-5.pp:$percentParallel.bs:128.nb:5.sml:80.tml:80",
      s"iwslt15.vi-en.both:false.back:false.bi_rnn:2:2.lstm:tanh.pairwise.w:512.r.a.dropout:0.2.ls:0.1.t:moses.c:moses.v:generated-simple-20000-5.pp:$percentParallel.bs:128.nb:5.sml:80.tml:80")
    case m => Seq(
      s"iwslt15.en-cs.en-de.en-fr.en-th.en-vi.both:true.back:true.bi_rnn:2:2.lstm:tanh.$m.l:8.w:512.r.a.dropout:0.2.ls:0.1.t:moses.c:moses.v:generated-simple-20000-5.pp:$percentParallel.bs:128.nb:5.sml:80.tml:80")
  }

  def main(args: Array[String]): Unit = {
//    val workingDir = File.currentWorkingDirectory / "temp" / "results"
//    workingDir.createIfNotExists(asDirectory = true)
//    val hostname = "gpu3.learning.cs.cmu.edu"
//    val ssh = new SSHClient()
//    ssh.useCompression()
//    ssh.loadKnownHosts()
//    ssh.connect(hostname)
//    try {
//      ssh.authPublickey("eplatani")
//      val results = Seq(
//        "P-512" -> Map(
//          "1%" -> iwslt15ExperimentDirectories("pairwise", 1),
//          "10%" -> iwslt15ExperimentDirectories("pairwise", 10)
//        ),
//        // "100%" -> iwslt15ExperimentDirectories("pairwise", 100)),
//        "HL-512-8-BT" -> Map(
//          "1%" -> iwslt15ExperimentDirectories("hyper_lang", 1),
//          "10%" -> iwslt15ExperimentDirectories("hyper_lang", 10)
//        ))
//      // "100%" -> iwslt15ExperimentDirectories("hyper_lang", 100)))
//      val parsedResults = results.flatMap(r => r._2.map(rr => {
//        (r._1, rr._1, rr._2.flatMap(d => {
//          (workingDir / d).createIfNotExists(asDirectory = true)
//          val localDestination = workingDir / d / "experiment.log"
//          val remotePath = s"~/code/symphony-mt/temp/experiments/$d/experiment.log"
//          ssh.newSCPFileTransfer().download(remotePath, new FileSystemFile(localDestination.toJava))
//          LogParser.parseEvaluationResults(localDestination)
//        }))
//      }))
//
//      ExperimentResults.plot(
//        results = parsedResults,
//        metric = BLEU,
//        datasets = Set("IWSLT-15"),
//        datasetTags = Set("tst2013"),
//        evalDatasets = Set("IWSLT-15"),
//        evalDatasetTags = Set("tst2012"),
//        title = "IWSLT-15")
//    } finally {
//      ssh.disconnect()
//    }

    val workingDir = File.currentWorkingDirectory / "temp" / "results"
    workingDir.createIfNotExists(asDirectory = true)
    val server = GoogleCloudServer
//     val remotePath = "~/code/symphony-mt/temp/experiments/iwslt15.en-cs.en-de.en-fr.en-th.en-vi.both:true.back:true.bi_rnn:2:2.lstm:tanh.hyper_lang.l:8.w:512.r.a.dropout:0.2.ls:0.1.t:moses.c:moses.v:generated-simple-20000-5.pp:100.bs:128.nb:5.sml:80.tml:80/experiment.log"
//    val remotePath = "~/code/symphony-mt/temp/experiments/iwslt15.en-cs.en-de.en-fr.en-th.en-vi.both:true.back:true.bi_rnn:2:2.lstm:tanh.hyper_lang.l:8.w:512.r.a.dropout:0.2.ls:0.1.t:moses.c:moses.v:generated-bpe-10000.pp:100.bs:128.nb:5.sml:80.tml:80/experiment.log"
//    val remotePath = "~/code/symphony-mt/temp/experiments/iwslt14.en-de.en-es.en-fr.en-he.en-it.en-nl.en-pt-br.en-ro.en-ru.both:true.back:true.bi_rnn:2:2.lstm:tanh.hyper_lang.l:8.w:512.r.a.dropout:0.2.ls:0.1.t:moses.c:moses.v:generated-bpe-10000.pp:100.bs:128.nb:5.sml:80.tml:80/experiment.log"
//    val remotePath = "~/code/symphony-mt/temp/experiments/iwslt17.both:true.back:true.bi_rnn:2:2.lstm:tanh.hyper_lang.l:8.w:512.r.a.dropout:0.2.ls:0.1.t:moses.c:moses.v:generated-bpe-10000.pp:100.bs:128.nb:5.sml:80.tml:80/experiment.log"

    val zsLanguagePairs = Set("it-ro", "ro-it", "nl-de", "de-nl")
    val languagePairs = Set("en", "de", "it", "nl", "ro").toSeq.combinations(2).map(p => s"${p(0)}-${p(1)}").toSet -- zsLanguagePairs
//    val remotePath = "~/code/symphony-mt/temp/experiments/iwslt15.en-cs.en-de.en-fr.en-th.en-vi.tw:true.ae:true.bi_rnn:2:2.lstm:tanh.google_multilingual.w:512.r.a.d:0.2.ls:0.1.t:moses.c:moses.v:simple-20000-5.pp:1.bs:128.nb:5.sml:50.tml:50/experiment.log"
//    val remotePath = "~/code/symphony-mt/temp/experiments/iwslt17.de:en.de:it.de:ro.en:it.en:nl.en:ro.it:nl.nl:ro.tw:true.ae:true.bi_rnn:2:2.lstm:tanh.hyper_lang:4.l:8.w:512.r.a.d:0.2.ls:0.1.t:moses.c:moses.v:simple-20000-5.pp:100.bs:128.nb:5.sml:50.tml:50/experiment.log"
    val remotePath = "~/../e_a_platanios/code/symphony-mt/temp/experiments/iwslt17.de:en.de:it.de:ro.en:it.en:nl.en:ro.it:nl.nl:ro.tw:true.ae:true.bi_rnn:2:2.lstm:tanh.hyper_lang:8.l:512.w:512.r.a.d:0.2.ls:0.1.t:moses.c:moses.v:simple-20000-5.pp:100.bs:128.nb:5.sml:50.tml:50/experiment.log"
//    val remotePath = "~/code/symphony-mt/temp/experiments/iwslt17.de-en.de-it.de-ro.en-it.en-nl.en-ro.it-nl.nl-ro.tw:true.ae:true.bi_rnn:2:2.lstm:tanh.hyper_lang.l:32.w:512.r.a.d:0.2.ls:0.1.t:moses.c:moses.v:simple-20000-5.pp:100.bs:128.nb:5.sml:50.tml:50/experiment.log"

    //    val languagePairs = Set("de", "ro").toSeq.combinations(2).map(p => s"${p(0)}-${p(1)}").toSet
//    val remotePath = "~/code/symphony-mt/temp/experiments/iwslt17.de-ro.tw:false.ae:false.bi_rnn:2:2.lstm:tanh.pairwise.w:512.r.a.d:0.2.ls:0.1.t:moses.c:moses.v:simple-20000-5.pp:100.bs:128.nb:5.sml:50.tml:50/experiment.log"
    val localDestination = workingDir / "test.log"
    val ssh = new SSHClient()
    ssh.useCompression()
    ssh.loadKnownHosts()
    ssh.connect(server.hostname)
    try {
      ssh.authPublickey(server.username)
      ssh.newSCPFileTransfer().download(remotePath, new FileSystemFile(localDestination.toJava))
      val bestResults = ExperimentResult.best(results = LogParser.parseEvaluationResults(localDestination))(
        metric = BLEU, datasets = Set("IWSLT-17"), datasetTags = Set("tst2017"), languagePairs = languagePairs)
      val mean = ExperimentResult.mean(results = bestResults)(
        metric = BLEU,
        datasets = Set("IWSLT-17"),
        datasetTags = Set("tst2017"),
        languagePairs = zsLanguagePairs)
      println(mean)
    } finally {
      ssh.disconnect()
    }
  }
}
