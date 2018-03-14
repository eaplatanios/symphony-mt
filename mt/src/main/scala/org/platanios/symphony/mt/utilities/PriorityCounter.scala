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

package org.platanios.symphony.mt.utilities

import scala.collection.mutable

/** Counter backed by a priority queue.
  *
  * This class allows counting elements, while providing constant time access to the element with the highest count at
  * any point in time. It is useful for algorithms similar to the one used to learn byte-pair-encodings of words, given
  * large text corpora.
  *
  * @author Emmanouil Antonios Platanios
  */
class PriorityCounter[A] extends scala.Cloneable {
  /** Internal array used by this priority counter to store all elements being counted. */
  protected val internalArray = new PriorityCounter.ResizableArrayAccess[(Long, A)]

  /** Map from element to location index in the internal array. */
  protected val locations = mutable.Map.empty[A, Int]

  /** Swaps the element stored at `index1` in the internal array, with that stored at `index2`. */
  protected def swap(index1: Int, index2: Int): Unit = {
    val element1 = cast(internalArray.p_array(index1))._2
    val element2 = cast(internalArray.p_array(index2))._2
    internalArray.p_swap(index1, index2)
    locations.update(element1, index2)
    locations.update(element2, index1)
  }

  // We do not use the first element of the array.
  internalArray.p_size0 += 1

  /** Returns the size of this counter (i.e., the number of elements it contains). */
  def size: Int = internalArray.length - 1

  /** Returns `true` if this counter does not contain any elements. */
  def isEmpty: Boolean = internalArray.p_size0 < 2

  /** Returns `true` if this counter is not empty. */
  def nonEmpty: Boolean = !isEmpty

  /** Casts an element of the internal array to the element type managed by this priority counter. */
  protected def cast(x: AnyRef): (Long, A) = x.asInstanceOf[(Long, A)]

  /** Fixes the order of the elements in `as`, upwards from position `m` (i.e., `m` to `0`).
    *
    * @return Boolean value indicating whether any swaps were made.
    */
  protected def fixUp(m: Int): Boolean = {
    var k: Int = m
    while (k > 1 && cast(internalArray.p_array(k / 2))._1 < cast(internalArray.p_array(k))._1) {
      swap(k, k / 2)
      k = k / 2
    }
    k != m
  }

  /** Fixes the order of the elements in `as`, downwards from position `m` until position `n`.
    *
    * @return Boolean value indicating whether any swaps were made.
    */
  protected def fixDown(m: Int, n: Int): Boolean = {
    var k: Int = m
    while (n >= 2 * k) {
      var j = 2 * k
      if (j < n && cast(internalArray.p_array(j))._1 < cast(internalArray.p_array(j + 1))._1)
        j += 1
      if (cast(internalArray.p_array(k))._1 >= cast(internalArray.p_array(j))._1) {
        return k != m
      } else {
        swap(k, j)
        k = j
      }
    }
    k != m
  }

  /** Updates the count for the provided element.
    *
    * @param  element Element whose count to update.
    * @param  count   Count value to use.
    * @return This priority counter.
    */
  def update(element: A, count: Long): this.type = {
    internalUpdate(element, count, (_) => count)
  }

  /** Adds the provided value to the count of the provided element.
    *
    * @param  element Element whose count to update.
    * @param  count   Count value to add.
    * @return This priority counter.
    */
  def add(element: A, count: Long): this.type = {
    internalUpdate(element, count, (location) => cast(internalArray.p_array(location))._1 + count)
  }

  /** Updates the count for the provided element.
    *
    * @param  element  Element whose count to update.
    * @param  count    Count value to use, if `element` is a new element.
    * @param  updateFn Function that takes the location of the element being updated and returns the count value to use.
    * @return This priority counter.
    */
  protected def internalUpdate(element: A, count: Long, updateFn: (Int) => Long): this.type = {
    val location = locations.getOrElseUpdate(element, internalArray.p_size0)
    if (location == internalArray.p_size0) {
      internalArray.p_ensureSize(internalArray.p_size0 + 1)
      internalArray.p_array(internalArray.p_size0) = (count, element).asInstanceOf[AnyRef]
      fixUp(internalArray.p_size0)
      internalArray.p_size0 += 1
    } else {
      internalArray.p_array(location) = (updateFn(location), element).asInstanceOf[AnyRef]
      fixUp(location)
      fixDown(location, internalArray.p_size0 - 1)
    }
    this
  }

  /** Returns the element with the highest count and removes it from this counter.
    *
    * @return Element with the highest count, along with that count.
    * @throws java.util.NoSuchElementException If there are no elements in this counter.
    */
  @throws[NoSuchElementException]
  def dequeueMax(): (Long, A) = {
    if (internalArray.p_size0 > 1) {
      internalArray.p_size0 -= 1
      val result = cast(internalArray.p_array(1))
      locations -= result._2
      val last = internalArray.p_array(internalArray.p_size0)
      internalArray.p_array(1) = last
      internalArray.p_array(internalArray.p_size0) = null
      val castedLast = cast(last)
      locations.update(castedLast._2, 1)
      fixDown(1, internalArray.p_size0 - 1)
      result
    } else
      throw new NoSuchElementException("The priority counter is empty.")
  }

  /** Returns the element with the highest count, without removing it from this counter.
    *
    * @return Element with the highest count, along with that count.
    * @throws java.util.NoSuchElementException If there are no elements in this counter.
    */
  def max: (Long, A) = {
    if (internalArray.p_size0 > 1)
      cast(internalArray.p_array(1))
    else
      throw new NoSuchElementException("The priority counter is empty.")
  }

  /** Removes all elements from this counter. After this operation is completed, the counter will be empty. */
  def clear(): Unit = {
    internalArray.p_size0 = 1
  }

  /** Clones this counter.
    *
    * @return A priority counter with the same elements as this one.
    */
  override def clone(): PriorityCounter[A] = {
    val pq = new PriorityCounter[A]
    val n = internalArray.p_size0
    pq.internalArray.p_ensureSize(n)
    java.lang.System.arraycopy(internalArray.p_array, 1, pq.internalArray.p_array, 1, n - 1)
    pq.internalArray.p_size0 = n
    pq.locations ++= locations.toSeq
    pq
  }
}

object PriorityCounter {
  def apply[A](): PriorityCounter[A] = new PriorityCounter[A]()

  /** Wrapper around the resizable array class that provides access to some of its protected fields. */
  protected[PriorityCounter] class ResizableArrayAccess[T]
      extends mutable.AbstractSeq[T]
          with mutable.ResizableArray[T]
          with Serializable {
    def p_size0: Int = size0
    def p_size0_=(s: Int): Unit = size0 = s
    def p_array: Array[AnyRef] = array
    def p_ensureSize(n: Int): Unit = super.ensureSize(n)
    def p_swap(a: Int, b: Int): Unit = super.swap(a, b)
  }
}
