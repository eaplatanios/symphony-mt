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

package org.platanios.symphony.mt.data

import org.platanios.symphony.mt.data.processors._
import org.platanios.symphony.mt.vocabulary._

import java.nio.file.{Path, Paths}

/**
  * @author Emmanouil Antonios Platanios
  */
case class DataConfig(
    // Loader
    workingDir: Path = Paths.get("working_dir"),
    loaderBufferSize: Int = 8192,
    loaderTokenizer: Tokenizer = MosesTokenizer(),
    loaderCleaner: Cleaner = MosesCleaner(),
    loaderVocab: DatasetVocabulary = GeneratedVocabulary(SimpleVocabularyGenerator(50000, -1, bufferSize = 8192)),
    // Corpus
    trainBatchSize: Long = 128,
    inferBatchSize: Long = 32,
    evaluateBatchSize: Long = 32,
    numBuckets: Int = 5,
    srcMaxLength: Int = 50,
    tgtMaxLength: Int = 50,
    bufferSize: Long = -1L,
    dropCount: Int = 0,
    numShards: Long = 1,
    shardIndex: Long = 0,
    numParallelCalls: Int = 4,
    // Vocabulary
    unknownToken: String = Vocabulary.UNKNOWN_TOKEN,
    beginOfSequenceToken: String = Vocabulary.BEGIN_OF_SEQUENCE_TOKEN,
    endOfSequenceToken: String = Vocabulary.END_OF_SEQUENCE_TOKEN)
