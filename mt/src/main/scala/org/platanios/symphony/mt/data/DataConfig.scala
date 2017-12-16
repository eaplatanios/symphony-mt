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

/**
  * @author Emmanouil Antonios Platanios
  */
case class DataConfig(
    // Corpus
    numBuckets: Int = 5,
    srcMaxLength: Int = 50,
    tgtMaxLength: Int = 50,
    srcReverse: Boolean = false,
    bufferSize: Long = -1L,
    dropCount: Int = 0,
    numShards: Int = 1,
    shardIndex: Int = 0,
    timeMajor: Boolean = false,
    numParallelCalls: Int = 4,
    // Vocabulary
    beginOfSequenceToken: String = Vocabulary.BEGIN_OF_SEQUENCE_TOKEN,
    endOfSequenceToken: String = Vocabulary.END_OF_SEQUENCE_TOKEN,
    unknownToken: String = Vocabulary.UNKNOWN_TOKEN)
