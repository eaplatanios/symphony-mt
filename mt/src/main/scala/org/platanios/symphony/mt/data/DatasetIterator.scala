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

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.io.data.InitializableIterator

import java.util.concurrent.atomic.AtomicBoolean

/** Iterator over a TensorFlow dataset.
  *
  * This iterator creates a TensorFlow dataset for `files` and a session that loads it and iterates over its elements.
  *
  * @param  files      Files from which to construct the dataset iterator.
  * @param  dataConfig Data configuration to use.
  *
  * @author Emmanouil Antonios Platanios
  */
class DatasetIterator protected (
    val files: LoadedDataset.GroupedFiles,
    val dataConfig: DataConfig
) extends Iterator[((Tensor, Tensor), (Tensor, Tensor))] {
  // TODO: [RECOVERY] Add ability to recover if the session crashes.

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
    if (initialized.compareAndSet(false, true)) {
      tf.createWith(graph) {
        session.run(targets = tf.initializers)
        session.run(targets = initOp)
      }
    }
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
