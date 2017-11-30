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

import org.platanios.symphony.mt.data.Vocabulary
import org.platanios.tensorflow.api.ops.training.optimizers.{GradientDescent, Optimizer}
import org.platanios.tensorflow.api.ops.training.optimizers.decay.Decay

import java.nio.file.{Path, Paths}

/**
  * @author Emmanouil Antonios Platanios
  */
case class Configuration(
    workingDir: Path = Paths.get("temp"),
    // Data
    dataNumBuckets: Int = 5,
    dataSrcMaxLength: Int = 50,
    dataTgtMaxLength: Int = 50,
    dataSrcReverse: Boolean = false,
    dataBufferSize: Long = -1L,
    dataDropCount: Int = 0,
    dataNumShards: Int = 1,
    dataShardIndex: Int = 0,
    dataTimeMajor: Boolean = false, // TODO: Currently unused.
    dataNumParallelCalls: Int = 4,
    // Vocabulary
    beginOfSequenceToken: String = Vocabulary.BEGIN_OF_SEQUENCE_TOKEN,
    endOfSequenceToken: String = Vocabulary.END_OF_SEQUENCE_TOKEN,
    unknownToken: String = Vocabulary.UNKNOWN_TOKEN,
    // Model
    modelNumUnits: Int = 128,
    modelDecodingMaxLengthFactor: Float = 2.0f,
    // Training
    trainBatchSize: Int = 128,
    trainMaxGradNorm: Float = 5.0f,
    trainNumSteps: Int = 12000,
    trainOptimizer: (Float, Decay) => Optimizer = GradientDescent(_, _, learningRateSummaryTag = "LearningRate"),
    trainLearningRateInitial: Float = 1.0f,
    trainLearningRateDecayRate: Float = 1.0f,
    trainLearningRateDecaySteps: Int = 10000,
    trainLearningRateDecayStartStep: Int = 0,
    trainSummarySteps: Int = 100,
    trainCheckpointSteps: Int = 1000,
    trainColocateGradientsWithOps: Boolean = true,
    // Inference
    inferBatchSize: Int = 32,
    inferBeamWidth: Int = 1,
    inferLengthPenaltyWeight: Float = 0.0f,
    // Logging
    logLossSteps: Int = 100,
    logEvalBatchSize: Int = 512,
    logTrainEvalSteps: Int = -1,
    logDevEvalSteps: Int = 1000,
    logTestEvalSteps: Int = 1000,
    // Miscellaneous
    logDevicePlacement: Boolean = false,
    randomSeed: Option[Int] = None,
    parallelIterations: Int = 10,
    swapMemory: Boolean = false,
    launchTensorBoard: Boolean = true)
