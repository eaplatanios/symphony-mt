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

package org.platanios.symphony.mt.data.utilities

import better.files._
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.utils.IOUtils

import java.io.InputStream
import java.util.zip.GZIPInputStream

/**
  * @author Emmanouil Antonios Platanios
  */
object CompressedFiles {
  def decompressTGZ(tgzFile: File, destination: File, bufferSize: Int = 8192): Unit = {
    decompressTGZStream(tgzFile.newInputStream, destination, bufferSize)
  }

  def decompressTar(tarFile: File, destination: File, bufferSize: Int = 8192): Unit = {
    decompressTarStream(tarFile.newInputStream, destination, bufferSize)
  }

  def decompressTGZStream(tgzStream: InputStream, destination: File, bufferSize: Int = 8192): Unit = {
    decompressTarStream(new GZIPInputStream(tgzStream), destination, bufferSize)
  }

  def decompressTarStream(tarStream: InputStream, destination: File, bufferSize: Int = 8192): Unit = {
    val inputStream = new TarArchiveInputStream(tarStream)
    var entry = inputStream.getNextTarEntry
    while (entry != null) {
      if (!entry.isDirectory) {
        val currentFile = destination.createChild(entry.getName)
        val parentFile = currentFile.parent
        if (!parentFile.exists)
          parentFile.createDirectories()
        IOUtils.copy(inputStream, currentFile.newFileOutputStream())
      }
      entry = inputStream.getNextTarEntry
    }
    inputStream.close()
  }
}
