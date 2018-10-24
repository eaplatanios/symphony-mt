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

import java.io.Serializable
import java.util.{PriorityQueue => JPriorityQueue}

import scala.collection.JavaConverters._
import scala.collection.generic.Growable

/** Bounded priority queue.
  *
  * This class wraps the original Java `PriorityQueue` class and modifies it such that only the top `maxSize` elements
  * are retained. The top `maxSize` elements are defined by an implicit `Ordering[A]`.
  *
  * @param  maxSize Number of elements to retain. If less than `1`, all elements are retained.
  * @param  ord     Implicit ordering to use for the elements.
  *
  * @author Emmanouil Antonios Platanios
  */
private[utilities] class BoundedPriorityQueue[A](maxSize: Int)(implicit ord: Ordering[A])
    extends Iterable[A] with Growable[A] with Serializable {

  private val underlying = {
    if (maxSize < 1) {
      new JPriorityQueue[A](ord)
    } else {
      new JPriorityQueue[A](maxSize, ord)
    }
  }

  override def iterator: Iterator[A] = {
    underlying.iterator.asScala
  }

  override def size: Int = {
    underlying.size
  }

  override def ++=(xs: TraversableOnce[A]): this.type = {
    xs.foreach(this += _)
    this
  }

  override def +=(elem: A): this.type = {
    if (maxSize < 1 || size < maxSize)
      underlying.offer(elem)
    else
      maybeReplaceLowest(elem)
    this
  }

  def poll(): A = {
    underlying.poll()
  }

  override def +=(elem1: A, elem2: A, elems: A*): this.type = {
    this += elem1 += elem2 ++= elems
  }

  override def clear(): Unit = {
    underlying.clear()
  }

  private def maybeReplaceLowest(a: A): Boolean = {
    val head = underlying.peek()
    if (head != null && ord.gt(a, head)) {
      underlying.poll()
      underlying.offer(a)
    } else {
      false
    }
  }
}

object BoundedPriorityQueue {
  def apply[A](maxSize: Int)(implicit ord: Ordering[A]): BoundedPriorityQueue[A] = {
    new BoundedPriorityQueue[A](maxSize)(ord)
  }
}
