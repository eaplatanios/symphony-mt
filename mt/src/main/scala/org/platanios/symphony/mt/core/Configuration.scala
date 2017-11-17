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

package org.platanios.symphony.mt.core

import java.nio.file.{Path, Paths}

import org.platanios.symphony.mt.data.Vocabulary

/**
  * @author Emmanouil Antonios Platanios
  */
case class Configuration(
    workingDir: Path = Paths.get("temp"),
    // Data
    batchSize: Int = 32,
    numBuckets: Int = 1,
    sourceMaxLength: Int = -1,
    targetMaxLength: Int = -1,
    sourceReverse: Boolean = false,
    // Vocabulary
    beginOfSequence: String = Vocabulary.BEGIN_OF_SEQUENCE_TOKEN,
    endOfSequence: String = Vocabulary.END_OF_SEQUENCE_TOKEN,
    unknownToken: String = Vocabulary.UNKNOWN_TOKEN,
    // Miscellaneous
    logDevicePlacement: Boolean = false,
    randomSeed: Option[Int] = None,
    parallelIterations: Int = 10,
    swapMemory: Boolean = false)
