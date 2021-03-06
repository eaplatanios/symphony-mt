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

/**
  * @author Emmanouil Antonios Platanios
  */
sealed trait DatasetType {
  override def toString: String
}

object DatasetType {
  def types: Set[DatasetType] = Set(Train, Dev, Test)
}

case object Train extends DatasetType {
  override def toString: String = "Train"
}

case object Dev extends DatasetType {
  override def toString: String = "Dev"
}

case object Test extends DatasetType {
  override def toString: String = "Test"
}
