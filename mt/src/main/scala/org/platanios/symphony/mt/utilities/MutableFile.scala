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

import better.files.File

/** Helper class used to represent mutable files. This class is useful for the data preprocessing steps.
  *
  * @param  file File being wrapped.
  *
  * @author Emmanouil Antonios Platanios
  */
private[mt] class MutableFile protected (protected var file: File) {
  def set(file: File): Unit = {
    this.file = file
  }

  def get: File = {
    file
  }
}

private[mt] object MutableFile {
  def apply(file: File): MutableFile = {
    new MutableFile(file)
  }
}
