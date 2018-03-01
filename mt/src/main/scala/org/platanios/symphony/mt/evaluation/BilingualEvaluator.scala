///* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
// *
// * Licensed under the Apache License, Version 2.0 (the "License"); you may not
// * use this file except in compliance with the License. You may obtain a copy of
// * the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// * License for the specific language governing permissions and limitations under
// * the License.
// */
//
//package org.platanios.symphony.mt.evaluation
//
//import org.platanios.symphony.mt.Language
//import org.platanios.symphony.mt.data._
//import org.platanios.symphony.mt.models.Model
//import org.platanios.tensorflow.api._
//
//// TODO: Is this really necessary now that we removed the translators interface?
//
//class BilingualEvaluator protected (
//    val metrics: Seq[MTMetric],
//    val srcLanguage: Language,
//    val tgtLanguage: Language,
//    val dataset: ParallelDataset
//) {
//  def evaluate(model: Model[_]): Map[String, Tensor] = {
//    val graph = Graph()
//    val session = Session(graph)
//    val values = tf.createWith(graph) {
//      val tfDataset = dataset.toTFBilingual(srcLanguage, tgtLanguage, repeat = false, isEval = true)
//      val iterator = tf.data.iteratorFromDataset(tfDataset)
//      val next = iterator.next()
//      val inputs = Seq(next._1._1, next._1._2)
//      val prediction = tf.callback(
//        (inputs: Seq[Tensor]) => {
//          val output = model.translate(
//            srcLanguage -> dataset.vocabulary(srcLanguage),
//            tgtLanguage -> dataset.vocabulary(tgtLanguage),
//            (inputs(0), inputs(1))).next()._2
//          Seq(output._1, output._2)
//        },
//        inputs, Seq(INT32, INT32))
//      val metricOps = metrics.map(_.streaming(((prediction(0), prediction(1)), next._2)))
//      val metricUpdateOps = tf.group(metricOps.map(_.update.op).toSet)
//      session.run(targets = tf.lookupsInitializer())
//      session.run(targets = Set(iterator.initializer, tf.localVariablesInitializer()))
//      try {
//        while (true)
//          session.run(targets = metricUpdateOps)
//        session.run(fetches = metricOps.map(_.value))
//      } catch {
//        case _: tf.OutOfRangeException =>
//          val value = session.run(fetches = metricOps.map(_.value))
//          session.close()
//          value
//      }
//    }
//    graph.close()
//    metrics.map(_.name).zip(values).toMap
//  }
//}
//
//object BilingualEvaluator {
//  def apply(
//      metrics: Seq[MTMetric],
//      srcLanguage: Language,
//      tgtLanguage: Language,
//      dataset: ParallelDataset
//  ): BilingualEvaluator = {
//    new BilingualEvaluator(metrics, srcLanguage, tgtLanguage, dataset)
//  }
//}
