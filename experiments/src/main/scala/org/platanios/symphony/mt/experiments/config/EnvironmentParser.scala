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

package org.platanios.symphony.mt.experiments.config

import org.platanios.symphony.mt.Environment

import com.typesafe.config.Config

import java.nio.file.Paths

/**
  * @author Emmanouil Antonios Platanios
  */
class EnvironmentParser(
  val experimentTag: String
) extends ConfigParser[Environment] {
  override def parse(config: Config): Environment = {
    Environment(
      workingDir = Paths.get(config.get[String]("working-dir")).resolve(experimentTag),
      allowSoftPlacement = config.get[Boolean]("allow-soft-placement"),
      logDevicePlacement = config.get[Boolean]("log-device-placement"),
      gpuAllowMemoryGrowth = config.get[Boolean]("gpu-allow-memory-growth"),
      useXLA = config.get[Boolean]("use-xla"),
      numGPUs = config.get[Int]("num-gpus"),
      parallelIterations = config.get[Int]("parallel-iterations"),
      swapMemory = config.get[Boolean]("swap-memory"),
      randomSeed = {
        val value = config.get[String]("random-seed")
        if (value == "none")
          None
        else
          Some(value.toInt)
      },
      traceSteps = {
        if (config.hasPath("trace-steps"))
          Some(config.get[Int]("trace-steps"))
        else
          None
      })
  }
}
