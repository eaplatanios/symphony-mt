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

package org.platanios.symphony.mt.core.hooks

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.client.{Executable, Fetchable}
import org.platanios.tensorflow.api.learn.Counter
import org.platanios.tensorflow.api.learn.hooks._

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.Path

/** Hooks that logs the perplexity value.
  *
  * @param  trigger      Hook trigger specifying when this hook is triggered (i.e., when it executes). If you only want
  *                      to log the tensor values at the end of a run and not during, then you should set `trigger` to
  *                      [[NoHookTrigger]] and `logAtEnd` to `true`.
  * @param  triggerAtEnd If `true`, this hook will be triggered at the end of the run. Note that if this flag is set to
  *                      `true`, then `tensors` must be computable without using a feed map for the `Session.run()`
  *                      call.
  * @param  formatter    Function used to format the message that is being logged. It takes the time taken since the
  *                      last logged message, the current step, and the current loss value, as input, and returns a
  *                      string to log.
  *
  * @author Emmanouil Antonios Platanios
  */
case class PerplexityLogger(
    log: Boolean = true,
    summaryDir: Path = null,
    trigger: HookTrigger = StepHookTrigger(1),
    triggerAtEnd: Boolean = true,
    average: Boolean = true,
    formatter: (Option[Double], Long, Float, Option[Double]) => String = null,
    summaryTag: String = "Perplexity"
) extends ModelDependentHook[
    (Tensor, Tensor), (Output, Output), (DataType, DataType), (Shape, Shape), (Output, Output),
    ((Tensor, Tensor), (Tensor, Tensor, Tensor)), ((Output, Output), (Output, Output, Output)),
    ((DataType, DataType), (DataType, DataType, DataType)), ((Shape, Shape), (Shape, Shape, Shape)),
    ((Output, Output), (Output, Output, Output))]
    with SummaryWriterHookAddOn {
  require(log || summaryDir != null, "At least one of 'log' and 'summaryDir' needs to be provided.")

  private[this] var step        : Variable = _
  private[this] var loss        : Output   = _
  private[this] var srcWordCount: Output   = _
  private[this] var tgtWordCount: Output   = _

  private[this] val internalTrigger  : HookTrigger = trigger.copy()
  private[this] var lastStep         : Long        = 0L
  private[this] var shouldTrigger    : Boolean     = false
  private[this] var totalLoss        : Float       = 0.0f
  private[this] var totalSrcWordCount: Long        = 0L
  private[this] var totalTgtWordCount: Long        = 0L

  override protected def begin(): Unit = {
    step = Counter.get(Graph.Keys.GLOBAL_STEP, local = false).getOrElse(throw new IllegalStateException(
      s"A ${Graph.Keys.GLOBAL_STEP.name} variable should be created in order to use the 'PerplexityLogger'."))
    internalTrigger.reset()
    shouldTrigger = false
    loss = modelInstance.loss.map(_.cast(FLOAT32))
        .flatMap(l => modelInstance.trainInput.map(o => l * tf.size(o._2._3))).orNull
    srcWordCount = modelInstance.trainInput.map(o => tf.sum(o._1._2)).orNull
    tgtWordCount = modelInstance.trainInput.map(o => tf.sum(o._2._3)).orNull
    totalLoss = 0.0f
    totalSrcWordCount = 0L
    totalTgtWordCount = 0L
  }

  override protected def afterSessionCreation(session: Session): Unit = {
    lastStep = session.run(fetches = step.value).scalar.asInstanceOf[Long]
  }

  override protected def beforeSessionRun[F, E, R](runContext: Hook.SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Option[Hook.SessionRunArgs[Seq[Output], Traversable[Op], Seq[Tensor]]] = {
    shouldTrigger = loss != null && internalTrigger.shouldTriggerForStep(lastStep.toInt)
    if (average || shouldTrigger)
      Some(Hook.SessionRunArgs(fetches = Seq(step.value, loss, srcWordCount, tgtWordCount)))
    else
      Some(Hook.SessionRunArgs(fetches = Seq(step.value)))
  }

  override protected def afterSessionRun[F, E, R](
      runContext: Hook.SessionRunContext[F, E, R],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor]]
  )(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Unit = {
    processFetches(runResult.values)
  }

  override protected def end(session: Session): Unit = {
    if (triggerAtEnd && lastStep.toInt != internalTrigger.lastTriggerStep().getOrElse(-1)) {
      shouldTrigger = true
      processFetches(session.run(fetches = Seq(step.value, loss, srcWordCount, tgtWordCount)))
    }
  }

  private[this] def processFetches(fetches: Seq[Tensor]): Unit = {
    lastStep = fetches.head.scalar.asInstanceOf[Long]
    if (average || shouldTrigger) {
      totalLoss += fetches(1).scalar.asInstanceOf[Float]
      totalSrcWordCount += fetches(2).scalar.asInstanceOf[Int]
      totalTgtWordCount += fetches(3).scalar.asInstanceOf[Int]
      if (shouldTrigger) {
        val totalWordCount = totalSrcWordCount + totalTgtWordCount
        val meanPerplexity = Math.exp(totalLoss / totalTgtWordCount).toFloat
        val elapsed = internalTrigger.updateLastTrigger(lastStep.toInt - 1).map(_._1)
        val message = {
          if (formatter != null) {
            formatter(elapsed, lastStep, meanPerplexity, elapsed.map(s => totalWordCount / (1000 * s)))
          } else {
            elapsed match {
              case Some(s) =>
                val wps = totalWordCount / (1000 * s)
                f"($s%9.3f s / $wps%5.2fk words/s ) Step: $lastStep%6d, Perplexity: $meanPerplexity%.4f"
              case None => f"(    timing not available yet ) Step: $lastStep%6d, Perplexity: $meanPerplexity%.4f"
            }
          }
        }
        if (log)
          PerplexityLogger.logger.info(message)
        writeSummary(lastStep, summaryTag, meanPerplexity)
        totalLoss = 0.0f
        totalSrcWordCount = 0L
        totalTgtWordCount = 0L
      }
    }
  }
}

object PerplexityLogger {
  private[PerplexityLogger] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Perplexity Logger"))
}
