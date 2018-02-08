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

package org.platanios.symphony.mt.translators.agents

import scala.collection.mutable

// TODO: Add support for circular buffer request mapper.

/** Request managers are used to keep track of what input was used for each request made to an agent/system. A hash map
  * can be used, for example, which guarantees that all inputs will always be "retrievable". However, if one requires
  * more efficiency, a circular buffer can be used instead. This means that old requests that have not been responded to
  * may be thrown away if the buffer fills up.
  *
  * @author Emmanouil Antonios Platanios
  */
private[agents] sealed trait RequestManager[R] {
  /** Sets the request data for the provided ID to the provided value.
    *
    * @param  id   Request ID.
    * @param  data Data to store for the request.
    */
  def set(id: Long, data: R): Unit

  /** Obtains the data corresponding to the provided request ID. `None` is returned if no data can be found for that ID.
    *
    * @param  id     Request ID.
    * @param  remove If `true`, the data is removed from this manager once returned.
    * @return Option containing the stored data for the provided request ID.
    */
  def get(id: Long, remove: Boolean = true): Option[R]

  /** Removes any data stored for the provided request ID, from this request manager.
    *
    * @param  id Request ID.
    */
  def remove(id: Long): Unit
}

/** Request manager that uses a hash map for storing the request data. */
private[agents] case class HashRequestManager[R] private[agents] () extends RequestManager[R] {
  /** Cache containing the stored request data. */
  private[this] val cache: mutable.LongMap[R] = mutable.LongMap.empty[R]

  /** Sets the request data for the provided ID to the provided value.
    *
    * @param  id   Request ID.
    * @param  data Data to store for the request.
    */
  override def set(id: Long, data: R): Unit = cache.update(id, data)

  /** Obtains the data corresponding to the provided request ID. `None` is returned if no data can be found for that ID.
    *
    * @param  id     Request ID.
    * @param  remove If `true`, the data is removed from this manager once returned.
    * @return Option containing the stored data for the provided request ID.
    */
  override def get(id: Long, remove: Boolean = true): Option[R] = {
    if (remove) cache.remove(id) else cache.get(id)
  }

  /** Removes any data stored for the provided request ID, from this request manager.
    *
    * @param  id Request ID.
    */
  override def remove(id: Long): Unit = cache.remove(id)
}

object RequestManager {
  /** Request manager type. */
  sealed trait Type {
    /** Creates a new request manager of this type. */
    def newManager[R](): RequestManager[R]
  }

  /** Request manager type that uses a hash map for storing the request data. */
  case object Hash extends Type {
    /** Creates a new request manager that uses a hash map for storing the request data. */
    override def newManager[R](): RequestManager[R] = HashRequestManager[R]()
  }
}
