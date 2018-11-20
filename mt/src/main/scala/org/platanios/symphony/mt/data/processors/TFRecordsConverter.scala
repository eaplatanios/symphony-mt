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

package org.platanios.symphony.mt.data.processors

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.data.newReader
import org.platanios.tensorflow.api.io.TFRecordWriter

import better.files._
import com.google.protobuf.ByteString
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import org.tensorflow.example._

import scala.collection.JavaConverters._
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
object TFRecordsConverter extends FileProcessor {
  private val logger         : Logger = Logger(LoggerFactory.getLogger("Data / TF Records Converter"))
  private val whitespaceRegex: Regex  = "\\s+".r

  override def process(file: File, language: Language): File = {
    val tfRecordsFile = convertedFile(file)
    if (tfRecordsFile.notExists) {
      logger.info(s"Converting file '$file' to TF records file '$tfRecordsFile'.")
      val reader = newReader(file)
      val writer = new TFRecordWriter(tfRecordsFile.path)
      reader.lines().toAutoClosedIterator.foreach(line => {
        writer.write(encodeSentenceAsTFExample(line, language))
      })
      writer.flush()
      writer.close()
      logger.info(s"Converted file '$file' to TF records file '$tfRecordsFile'.")
    }
    tfRecordsFile
  }

  protected def convertedFile(originalFile: File): File = {
    originalFile.sibling(originalFile.name + ".tfrecords")
  }

  protected def encodeSentenceAsTFExample(sentence: String, language: Language): Example = {
    val processedSentence = preprocessSentence(sentence, language)
    Example.newBuilder()
        .setFeatures(
          Features.newBuilder()
              .putFeature(
                "sentence",
                Feature.newBuilder()
                    .setBytesList(
                      BytesList.newBuilder()
                          .addAllValue(processedSentence.map(ByteString.copyFromUtf8).asJava))
                    .build())
              .putFeature(
                "length",
                Feature.newBuilder()
                    .setInt64List(Int64List.newBuilder().addValue(processedSentence.length))
                    .build()))
        .build()
  }

  protected def preprocessSentence(sentence: String, language: Language): Seq[String] = {
    whitespaceRegex.split(sentence)
  }
}
