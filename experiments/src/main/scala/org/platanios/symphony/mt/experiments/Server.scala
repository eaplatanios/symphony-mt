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

package org.platanios.symphony.mt.experiments

/**
  * @author Emmanouil Antonios Platanios
  */
trait Server {
  val hostname: String
  val username: String
}

case object GPU3Server extends Server {
  override val hostname: String = "gpu3.learning.cs.cmu.edu"
  override val username: String = "eplatani"
}

case object GoogleCloudServer extends Server {
  override val hostname: String = "104.155.134.36"
  override val username: String = "e.a.platanios"
}
