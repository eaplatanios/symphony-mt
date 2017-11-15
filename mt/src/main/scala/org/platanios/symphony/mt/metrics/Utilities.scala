/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.symphony.mt.metrics

import scala.collection.mutable

/** Contains helper methods for computing machine translation metrics.
  *
  * @author Emmanouil Antonios Platanios
  */
object Utilities {
  /** Extracts and counts the number of occurrences of all n-grams up to a provided maximum order from the provided
    * input sequence.
    *
    * @param  sequence Sequence from which to extract n-grams.
    * @param  maxOrder Maximum n-gram order.
    * @return Map from n-gram to count.
    */
  def countNGrams[T](sequence: Seq[T], maxOrder: Int): Map[Seq[T], Long] = {
    val counter = mutable.HashMap.empty[Seq[T], Long].withDefaultValue(0L)
    (1 to maxOrder).foreach(order => sequence.iterator.sliding(order).withPartial(false).foreach(counter(_) += 1L))
    counter.toMap
  }
}
