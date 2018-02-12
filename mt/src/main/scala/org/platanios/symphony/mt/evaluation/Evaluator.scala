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

package org.platanios.symphony.mt.evaluation

import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.translators.Translator
import org.platanios.tensorflow.api._

class Evaluator protected () {
  def evaluate(
      metric: MTMetric,
      translator: Translator,
      datasetFiles: LoadedDataset.GroupedFiles,
      datasetType: DatasetType,
      dataConfig: DataConfig
  ): Tensor = {
    val srcLang = datasetFiles.srcLang
    val tgtLang = datasetFiles.tgtLang
    val graph = Graph()
    val session = Session(graph)
    val value = tf.createWith(graph) {
      val dataset = datasetFiles.createTrainDataset(datasetType, repeat = false, dataConfig, isEval = true)
      val iterator = tf.data.iteratorFromDataset(dataset)
      val next = iterator.next()
      val inputs = Seq(next._1._1, next._1._2)
      val prediction = tf.callback(
        (inputs: Seq[Tensor]) => {
          val output = translator.translate(srcLang, tgtLang, (inputs(0), inputs(1)))
          Seq(output._1, output._2)
        },
        inputs, Seq(INT32, INT32))
      val metricOps = metric.streaming(((prediction(0), prediction(1)), next._2))
      session.run(targets = tf.initializers)
      session.run(targets = Set(iterator.initializer, tf.localVariablesInitializer()))
      try {
        while (true)
          session.run(targets = metricOps.update)
        session.run(fetches = metricOps.value)
      } catch {
        case _: tf.OutOfRangeException =>
          val value = session.run(fetches = metricOps.value)
          session.close()
          value
      }
    }
    graph.close()
    value
  }
}

object Evaluator {
  def apply(): Evaluator = {
    new Evaluator()
  }
}
