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

package org.platanios.symphony.mt

import org.platanios.tensorflow.api.{DataType, Output, Shape, Tensor, tf}

package object data {
  type TFMonolingualDataset = tf.data.Dataset[(Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape)]

  type TFBilingualDataset = tf.data.Dataset[
      ((Tensor, Tensor), (Tensor, Tensor)),
      ((Output, Output), (Output, Output)),
      ((DataType, DataType), (DataType, DataType)),
      ((Shape, Shape), (Shape, Shape))]

  private[data] def joinMonolingualDatasets(
      datasets: Seq[tf.data.Dataset[Tensor, Output, DataType, Shape]]
  ): tf.data.Dataset[Tensor, Output, DataType, Shape] = {
    datasets.reduce((d1, d2) => d1.concatenate(d2))
  }

  private[data] def joinBilingualDatasets(
      datasets: Seq[tf.data.Dataset[(Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape)]]
  ): tf.data.Dataset[(Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape)] = {
    datasets.reduce((d1, d2) => d1.concatenate(d2))
  }
}
