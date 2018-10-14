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

package org.platanios.symphony.mt.vocabulary

import org.platanios.symphony.mt.data.DataConfig

import better.files.File

/** Represents a vocabulary of coded words (e.g., using the byte-pair-encoding method).
  *
  * @param  file                 File containing the vocabulary, with one word per line.
  * @param  size                 Size of this vocabulary (i.e., number of words).
  * @param  encoder              Sentence encoding function (each sentence is represented as a sequence of words).
  * @param  decoder              Sentence decoding function (each sentence is represented as a sequence of words).
  * @param  unknownToken         Token representing unknown symbols (i.e., not included in this vocabulary).
  * @param  beginOfSequenceToken Token representing the beginning of a sequence.
  * @param  endOfSequenceToken   Token representing the end of a sequence.
  *
  * @author Emmanouil Antonios Platanios
  */
class CodedVocabulary protected (
    override val file: File,
    override val size: Int,
    protected val encoder: Seq[String] => Seq[String],
    protected val decoder: Seq[String] => Seq[String],
    override val unknownToken: String,
    override val beginOfSequenceToken: String,
    override val endOfSequenceToken: String
) extends Vocabulary(file, size, unknownToken, beginOfSequenceToken, endOfSequenceToken) {
  /** Encodes the provided sequence using this vocabulary. This is typically an identity function.
    *
    * This method is useful for coded vocabularies, such as the byte-pair-encoding vocabulary.
    *
    * @param  sequence Sequence of tokens to encode.
    * @return Encoded sequence of tokens that may differ in size from the input sequence.
    */
  override def encodeSequence(sequence: Seq[String]): Seq[String] = {
    encoder(sequence)
  }

  /** Decodes the provided sequence using this vocabulary. This is typically an identity function.
    *
    * This method is useful for coded vocabularies, such as the byte-pair-encoding vocabulary.
    *
    * @param  sequence Sequence of tokens to decode.
    * @return Decoded sequence of tokens that may differ in size from the input sequence.
    */
  override def decodeSequence(sequence: Seq[String]): Seq[String] = {
    decoder(sequence)
  }
}

object CodedVocabulary {
  /** Creates a new coded vocabulary.
    *
    * @param  file                 File containing the vocabulary, with one word per line.
    * @param  size                 Size of this vocabulary (i.e., number of words).
    * @param  encoder              Sentence encoding function (each sentence is represented as a sequence of words).
    * @param  decoder              Sentence decoding function (each sentence is represented as a sequence of words).
    * @param  unknownToken         Token representing unknown symbols (i.e., not included in this vocabulary).
    * @param  beginOfSequenceToken Token representing the beginning of a sequence.
    * @param  endOfSequenceToken   Token representing the end of a sequence.
    * @return Created vocabulary.
    */
  protected def apply(
      file: File,
      size: Int,
      encoder: Seq[String] => Seq[String],
      decoder: Seq[String] => Seq[String],
      unknownToken: String,
      beginOfSequenceToken: String,
      endOfSequenceToken: String
  ): CodedVocabulary = {
    new CodedVocabulary(file, size, encoder, decoder, unknownToken, beginOfSequenceToken, endOfSequenceToken)
  }

  /** Creates a new coded vocabulary from the provided vocabulary file.
    *
    * The method first checks if the specified vocabulary file exists and if it does, it checks that special tokens are
    * being used correctly. If not, this method can optionally create a new file by prepending the appropriate tokens to
    * the existing one.
    *
    * The special tokens check simply involves checking whether the first three tokens in the vocabulary file match the
    * specified `unknownToken`, `beginSequenceToken`, and `endSequenceToken` values.
    *
    * @param  file               Vocabulary file to check.
    * @param  encoder            Sentence encoding function (each sentence is represented as a sequence of words).
    * @param  decoder            Sentence decoding function (each sentence is represented as a sequence of words).
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
      file: File,
      encoder: Seq[String] => Seq[String],
      decoder: Seq[String] => Seq[String],
      checkSpecialTokens: Boolean = true,
      directory: File = null,
      dataConfig: DataConfig = DataConfig()
  ): CodedVocabulary = {
    val check = Vocabulary.check(
      file, checkSpecialTokens, directory,
      dataConfig.unknownToken, dataConfig.beginOfSequenceToken, dataConfig.endOfSequenceToken)
    check match {
      case None => throw new IllegalArgumentException(s"Could not load the vocabulary file located at '$file'.")
      case Some((size, path)) => CodedVocabulary(
        path, size, encoder, decoder,
        dataConfig.unknownToken, dataConfig.beginOfSequenceToken, dataConfig.endOfSequenceToken)
    }
  }
}
