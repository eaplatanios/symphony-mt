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

package org.platanios.symphony.mt.translators.actors

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.data.{DataConfig, LoadedDataset, TRAIN_DATASET}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.io.data.InitializableIterator

import akka.actor.ActorRef

import java.util.concurrent.atomic.AtomicBoolean

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class TrainScheduler(
    protected val dataset: LoadedDataset,
    protected val agents: Map[Language, ActorRef]
) {
  /** Initializes this train scheduler. This method is always called by the translation system, in order to start
    * the training process. */
  protected def initialize(): Unit

  /** Responds to a translation agent's train response. This method is called by the translation system, whenever it
    * receives an agent train response message. */
  protected def onTrainResponse(agent: ActorRef): Unit
}

object TrainScheduler {
  class DatasetIterator protected (
      val files: LoadedDataset.GroupedFiles,
      val dataConfig: DataConfig
  ) extends Iterator[((Tensor, Tensor), (Tensor, Tensor))] {
    protected val graph  : Graph   = Graph()
    protected val session: Session = Session(graph)

    protected val iterator: InitializableIterator[
        ((Tensor, Tensor), (Tensor, Tensor)),
        ((Output, Output), (Output, Output)),
        ((DataType, DataType), (DataType, DataType)),
        ((Shape, Shape), (Shape, Shape))] = tf.createWith(graph) {
      tf.data.iteratorFromDataset(
        files.createTrainDataset(TRAIN_DATASET, repeat = true, dataConfig, isEval = false))
    }

    protected val initOp    : Op                                   = tf.createWith(graph)(iterator.initializer)
    protected val nextOutput: ((Output, Output), (Output, Output)) = tf.createWith(graph)(iterator.next())

    protected var initialized: AtomicBoolean = new AtomicBoolean(false)

    override def hasNext: Boolean = true
    override def next(): ((Tensor, Tensor), (Tensor, Tensor)) = {
      initialized.compareAndSet(false, {
        session.run(targets = initOp)
        true
      })
      session.run(fetches = nextOutput)
    }
  }

  object DatasetIterator {
    def apply(
        files: LoadedDataset.GroupedFiles,
        dataConfig: DataConfig
    ): DatasetIterator = {
      new DatasetIterator(files, dataConfig)
    }
  }
}
