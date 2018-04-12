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

package org.platanios.symphony.mt.experiments

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

/**
  *
  * Example command:
  * {{{
  *   sbt "mt/runMain org.platanios.symphony.mt.experiments.Experiment
  *     --task train
  *     --working-dir temp/experiments
  *     --data-dir temp/data
  *     --dataset iwslt15
  *     --language-pairs en:cs,en:de,en:fr,en:th,en:vi,en:zh
  *     --eval-datasets dev2010,tst2011,tst2012,tst2013
  *     --tokenizer moses
  *     --cleaner moses
  *     --vocabulary bpe:10000
  *     --batch-size 128
  *     --num-buckets 5
  *     --src-max-length 50
  *     --tgt-max-length 50
  *     --buffer-size 1024
  *     --model-arch bi_rnn:4:4
  *     --model-cell lstm:tanh
  *     --model-type hyper_lang
  *     --word-embed-size 256
  *     --lang-embed-size 16
  *     --residual
  *     --dropout 0.2
  *     --label-smoothing 0.1
  *     --beam-width 10
  *     --opt amsgrad:0.001
  *     --opt-max-norm 100.0
  *     --num-steps 100000
  *     --summary-steps 100
  *     --checkpoint-steps 1000
  *     --log-loss-steps 100
  *     --log-eval-steps 5000
  *     --launch-tensorboard
  *     --tensorboard-host localhost
  *     --tensorboard-port 6006
  *     --num-gpus 1
  *     --seed 10"
  * }}}
  *
  * @author Emmanouil Antonios Platanios
  */
object Experiment extends App {
  private[experiments] val logger = Logger(LoggerFactory.getLogger("Experiment"))

  val experimentConfig = ExperimentConfig.parser.parse(args, ExperimentConfig()).get

  experimentConfig.initialize()
  experimentConfig.logSummary()
  experimentConfig.run()
}
