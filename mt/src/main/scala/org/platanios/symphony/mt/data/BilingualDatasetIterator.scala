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
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.io.data.InitializableIterator

import java.util.concurrent.atomic.AtomicBoolean

/** Iterator over pairs of sentences in a TensorFlow dataset, for the provided languages.
  *
  * This iterator creates a TensorFlow dataset for `dataset` and a session that loads it and iterates over its elements.
  *
  * @param  dataset    Dataset from which to construct this iterator.
  * @param  language1  First language.
  * @param  language2  Second language.
  * @param  dataConfig Data configuration to use.
  *
  * @author Emmanouil Antonios Platanios
  */
class BilingualDatasetIterator[T <: ParallelDataset[T]] protected (
    val dataset: ParallelDataset[T],
    val language1: Language,
    val language2: Language,
    val dataConfig: DataConfig,
    val repeat: Boolean = true,
    val isEval: Boolean = false
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
      dataset.filterTypes(Train).toTFBilingual(language1, language2, dataConfig, repeat, isEval))
  }

  protected val initOp    : Op                                   = tf.createWith(graph)(iterator.initializer)
  protected val nextOutput: ((Output, Output), (Output, Output)) = tf.createWith(graph)(iterator.next())

  protected var initialized: AtomicBoolean = new AtomicBoolean(false)

  override def hasNext: Boolean = true
  override def next(): ((Tensor, Tensor), (Tensor, Tensor)) = {
    if (initialized.compareAndSet(false, true)) {
      tf.createWith(graph) {
        session.run(targets = tf.lookupsInitializer())
        session.run(targets = initOp)
      }
    }
    session.run(fetches = nextOutput)
  }
}

object BilingualDatasetIterator {
  def apply[T <: ParallelDataset[T]](
      dataset: ParallelDataset[T],
      language1: Language,
      language2: Language,
      dataConfig: DataConfig,
      repeat: Boolean = true,
      isEval: Boolean = false
  ): BilingualDatasetIterator[T] = {
    new BilingualDatasetIterator[T](dataset, language1, language2, dataConfig, repeat, isEval)
  }
}
