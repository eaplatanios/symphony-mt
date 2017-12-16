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

package org.platanios.symphony.mt.data

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.lookup.LookupTable

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

import scala.collection.mutable

/** Contains utilities for dealing with vocabulary files.
  *
  * @author Emmanouil Antonios Platanios
  */
case class Vocabulary(path: Path, size: Int, dataConfig: DataConfiguration) {
  /** Creates a vocabulary lookup table (from word string to word ID), from the provided vocabulary file.
    *
    * @return Vocabulary lookup table.
    */
  def lookupTable(): LookupTable = {
    val table = tf.indexTableFromFile(path.toAbsolutePath.toString, defaultValue = Vocabulary.UNKNOWN_TOKEN_ID)
    table.initialize()
    table
  }
}

/** Contains utilities for dealing with vocabularies. */
object Vocabulary {
  /** Creates a new vocabulary from the provided vocabulary file.
    *
    * The method first checks if the specified vocabulary file exists and if it does, it checks that special tokens are
    * being used correctly. If not, this method can optionally create a new file by prepending the appropriate tokens to
    * the existing one.
    *
    * The special tokens check simply involves checking whether the first three tokens in the vocabulary file match the
    * specified `unknownToken`, `beginSequenceToken`, and `endSequenceToken` values.
    *
    * @param  file               Vocabulary file to check.
    * @param  checkSpecialTokens Boolean value indicating whether or not to check for the use of special tokens, and
    *                            prepend them while creating a new vocabulary file, if the check fails.
    * @param  directory          Directory to use when creating the new vocabulary file, in case the special tokens
    *                            check fails. Defaults to the current directory in which `file` is located, meaning
    *                            that if the special tokens check fails, `file` will be replaced with the appended
    *                            vocabulary file.
    * @param  dataConfig         Data configuration that includes information about the special tokens.
    * @return Constructed vocabulary.
    * @throws IllegalArgumentException If the provided vocabulary file could not be loaded.
    */
  @throws[IllegalArgumentException]
  def apply(
      file: Path, checkSpecialTokens: Boolean = true, directory: Path = null,
      dataConfig: DataConfiguration = DataConfiguration()
  ): Vocabulary = {
    val check = Vocabulary.check(
      file, checkSpecialTokens, directory,
      dataConfig.beginOfSequenceToken, dataConfig.endOfSequenceToken, dataConfig.unknownToken)
    check match {
      case None => throw new IllegalArgumentException(s"Could not load the vocabulary file located at '$file'.")
      case Some((size, path)) => Vocabulary(path, size, dataConfig)
    }
  }

  private[this] val logger: Logger = Logger(LoggerFactory.getLogger("Vocabulary"))

  val BEGIN_OF_SEQUENCE_TOKEN: String = "<s>"
  val END_OF_SEQUENCE_TOKEN  : String = "</s>"
  val UNKNOWN_TOKEN          : String = "<unk>"
  val UNKNOWN_TOKEN_ID       : Int    = 0

  /** Checks if the specified vocabulary file exists and if it does, checks that special tokens are being used
    * correctly. If not, this method can optionally create a new file by prepending the appropriate tokens to the
    * existing one.
    *
    * The special tokens check simply involves checking whether the first three tokens in the vocabulary file match the
    * specified `unknownToken`, `beginSequenceToken`, and `endSequenceToken` values.
    *
    * @param  file                 Vocabulary file to check.
    * @param  checkSpecialTokens   Boolean value indicating whether or not to check for the use of special tokens, and
    *                              prepend them while creating a new vocabulary file, if the check fails.
    * @param  directory            Directory to use when creating the new vocabulary file, in case the special tokens
    *                              check fails. Defaults to the current directory in which `file` is located, meaning
    *                              that if the special tokens check fails, `file` will be replaced with the appended
    *                              vocabulary file.
    * @param  beginOfSequenceToken Special token for the beginning of a sequence. Defaults to `<s>`.
    * @param  endOfSequenceToken   Special token for the end of a sequence. Defaults to `</s>`.
    * @param  unknownToken         Special token for unknown tokens. Defaults to `<unk>`.
    */
  private[Vocabulary] def check(
      file: Path, checkSpecialTokens: Boolean = true, directory: Path = null,
      beginOfSequenceToken: String = BEGIN_OF_SEQUENCE_TOKEN, endOfSequenceToken: String = END_OF_SEQUENCE_TOKEN,
      unknownToken: String = UNKNOWN_TOKEN): Option[(Int, Path)] = {
    if (!Files.exists(file)) {
      None
    } else {
      logger.info(s"Vocabulary file '$file' exists.")
      val reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)
      val tokens = mutable.ListBuffer.empty[String]
      var line = reader.readLine()
      while (line != null) {
        tokens ++= line.split("\\s+").toSeq
        line = reader.readLine()
      }
      reader.close()
      if (!checkSpecialTokens) {
        Some((tokens.size, file))
      } else {
        // Verify that the loaded vocabulary using the right special tokens.
        // If it does not, use those tokens and generate a new vocabulary file.
        assert(tokens.lengthCompare(3) >= 0, "The loaded vocabulary must contain at least three tokens.")
        if (tokens(0) != unknownToken || tokens(1) != beginOfSequenceToken || tokens(2) != endOfSequenceToken) {
          logger.info(
            s"The first 3 vocabulary tokens [${tokens(0)}, ${tokens(1)}, ${tokens(2)}] " +
                s"are not equal to [$unknownToken, $beginOfSequenceToken, $endOfSequenceToken].")
          tokens.prepend(unknownToken, beginOfSequenceToken, endOfSequenceToken)
          val newFile = if (directory != null) directory.resolve(file.getFileName) else file
          val writer = Files.newBufferedWriter(newFile, StandardCharsets.UTF_8)
          tokens.foreach(token => writer.write(s"$token\n"))
          writer.close()
          Some((tokens.size, newFile))
        } else {
          Some((tokens.size, file))
        }
      }
    }
  }

  /** Creates vocabulary lookup tables (from word string to word ID), from the provided vocabulary files.
    *
    * @param  srcFile Source vocabulary file.
    * @param  tgtFile Target vocabulary file.
    * @return Tuple contain the source vocabulary lookup table and the target one.
    */
  private[Vocabulary] def createTables(srcFile: Path, tgtFile: Path): (tf.LookupTable, tf.LookupTable) = {
    val srcPath = srcFile.toAbsolutePath.toString
    val tgtPath = tgtFile.toAbsolutePath.toString
    val sourceTable = tf.indexTableFromFile(srcPath, defaultValue = UNKNOWN_TOKEN_ID)
    val targetTable = {
      if (srcFile == tgtFile)
        sourceTable
      else
        tf.indexTableFromFile(tgtPath, defaultValue = UNKNOWN_TOKEN_ID)
    }
    sourceTable.initialize()
    if (srcFile != tgtFile)
      targetTable.initialize()
    (sourceTable, targetTable)
  }
}
