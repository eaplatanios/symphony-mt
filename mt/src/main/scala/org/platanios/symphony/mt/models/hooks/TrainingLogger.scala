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

package org.platanios.symphony.mt.models.hooks

import org.platanios.symphony.mt.models.{Sentences, SentencesWithLanguage, SentencesWithLanguagePair}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToTensor}
import org.platanios.tensorflow.api.learn.Counter
import org.platanios.tensorflow.api.learn.hooks._

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.nio.file.Path

/** Hooks that logs the training speed, the perplexity value, and the gradient norm while training.
  *
  * @param  trigger      Hook trigger specifying when this hook is triggered (i.e., when it executes). If you only want
  *                      to log the tensor values at the end of a run and not during, then you should set `trigger` to
  *                      [[NoHookTrigger]] and `logAtEnd` to `true`.
  * @param  triggerAtEnd If `true`, this hook will be triggered at the end of the run. Note that if this flag is set to
  *                      `true`, then `tensors` must be computable without using a feed map for the `Session.run()`
  *                      call.
  * @param  formatter    Function used to format the message that is being logged. It takes the time taken since the
  *                      last logged message, the current step, the current gradients norm, the current perplexity
  *                      value, as input, and returns a string to log.
  *
  * @author Emmanouil Antonios Platanios
  */
case class TrainingLogger(
    log: Boolean = true,
    summaryDir: Path = null,
    trigger: HookTrigger = StepHookTrigger(1),
    triggerAtEnd: Boolean = true,
    average: Boolean = true,
    dataParallelFactor: Float = 1.0f,
    formatter: (Option[Double], Long, Float, Float, Option[Double]) => String = null,
    summaryTag: String = "Perplexity"
) extends ModelDependentHook[
    /* In       */ SentencesWithLanguagePair[String],
    /* TrainIn  */ (SentencesWithLanguagePair[String], Sentences[String]),
    /* Out      */ SentencesWithLanguage[String],
    /* TrainOut */ SentencesWithLanguage[Float],
    /* Loss     */ Float,
    /* EvalIn   */ (SentencesWithLanguage[String], (SentencesWithLanguagePair[String], Sentences[String]))]
    with SummaryWriterHookAddOn {
  require(log || summaryDir != null, "At least one of 'log' and 'summaryDir' needs to be provided.")

  private var step         : Variable[Long] = _
  private var gradientsNorm: Output[Float]  = _
  private var loss         : Output[Float]  = _
  private var srcWordCount : Output[Long]   = _
  private var tgtWordCount : Output[Long]   = _

  private val internalTrigger   : HookTrigger = trigger.copy()
  private var lastStep          : Long        = 0L
  private var shouldTrigger     : Boolean     = false
  private var totalGradientsNorm: Float       = 0.0f
  private var totalLoss         : Float       = 0.0f
  private var totalSrcWordCount : Long        = 0L
  private var totalTgtWordCount : Long        = 0L

  override protected def begin(): Unit = {
    step = Counter.get(Graph.Keys.GLOBAL_STEP, local = false).getOrElse(throw new IllegalStateException(
      s"A ${Graph.Keys.GLOBAL_STEP.name} variable should be created in order to use the 'TrainingLogger'."))
    internalTrigger.reset()
    shouldTrigger = false
    gradientsNorm = modelInstance.gradientsAndVariables.map(g => tf.globalNorm(g.map(_._1))).orNull
    loss = modelInstance.loss.map(_.toFloat).flatMap(l => {
      modelInstance.trainInput.map(o => l * /* batch size */ tf.size(o._2._2).toFloat)
    }).orNull
    srcWordCount = modelInstance.trainInput.map(o => {
      /* source sentence lengths */ tf.sum(o._1._3._2).toLong
    }).orNull
    tgtWordCount = modelInstance.trainInput.map(o => {
      /* target sentence lengths */ tf.sum(o._2._2).toLong + /* batch size */ tf.size(o._2._2)
    }).orNull
    totalLoss = 0.0f
    totalSrcWordCount = 0L
    totalTgtWordCount = 0L
  }

  override protected def afterSessionCreation(session: Session): Unit = {
    lastStep = session.run(fetches = step.value).scalar
  }

  override protected def beforeSessionRun[C: OutputStructure, CV](
      runContext: Hook.SessionRunContext[C, CV]
  )(implicit
      evOutputToTensorC: OutputToTensor.Aux[C, CV]
  ): Option[Hook.SessionRunArgs[Seq[Output[Any]], Seq[Tensor[Any]]]] = {
    shouldTrigger = gradientsNorm != null && loss != null && internalTrigger.shouldTriggerForStep(lastStep.toInt + 1)
    if (average || shouldTrigger)
      Some(Hook.SessionRunArgs(fetches = Seq[Output[Any]](step.value, gradientsNorm, loss, srcWordCount, tgtWordCount)))
    else
      Some(Hook.SessionRunArgs(fetches = Seq[Output[Any]](step.value)))
  }

  override protected def afterSessionRun[C: OutputStructure, CV](
      runContext: Hook.SessionRunContext[C, CV],
      runResult: Hook.SessionRunResult[Seq[Tensor[Any]]]
  )(implicit
      evOutputToTensorC: OutputToTensor.Aux[C, CV]
  ): Unit = {
    processFetches(runResult.result)
  }

  override protected def end(session: Session): Unit = {
    if (triggerAtEnd && lastStep.toInt != internalTrigger.lastTriggerStep().getOrElse(-1)) {
      shouldTrigger = true
      processFetches(session.run(fetches = Seq[Output[Any]](
        step.value, gradientsNorm, loss, srcWordCount, tgtWordCount)))
    }
  }

  private def processFetches(fetches: Seq[Tensor[Any]]): Unit = {
    lastStep = fetches.head.scalar.asInstanceOf[Long]
    if (average || shouldTrigger) {
      totalGradientsNorm += fetches(1).scalar.asInstanceOf[Float]
      totalLoss += fetches(2).scalar.asInstanceOf[Float]
      totalSrcWordCount += fetches(3).scalar.asInstanceOf[Long].toInt
      totalTgtWordCount += fetches(4).scalar.asInstanceOf[Long].toInt
      if (shouldTrigger) {
        val numSteps = (lastStep * dataParallelFactor).toInt
        val elapsed = internalTrigger.updateLastTrigger(lastStep.toInt)
        val elapsedTime = elapsed.map(_._1)
        val totalWordCount = (totalSrcWordCount + totalTgtWordCount) * dataParallelFactor
        val meanGradientsNorm = totalGradientsNorm / elapsed.map(_._2).getOrElse(1)
        val meanPerplexity = Math.exp(totalLoss / totalTgtWordCount).toFloat
        val message = {
          if (formatter != null) {
            formatter(
              elapsedTime, numSteps, meanGradientsNorm, meanPerplexity,
              elapsedTime.map(s => totalWordCount / (1000 * s)))
          } else {
            elapsedTime match {
              case Some(s) =>
                val wps = totalWordCount / (1000 * s)
                f"($s%9.3f s / $wps%5.2fk words/s ) " +
                    f"Step: $numSteps%6d, " +
                    f"Perplexity: $meanPerplexity%12.4f, " +
                    f"Gradients Norm: $meanGradientsNorm%12.4f"
              case None =>
                f"(    timing not available yet ) " +
                    f"Step: $numSteps%6d, " +
                    f"Perplexity: $meanPerplexity%12.4f, " +
                    f"Gradients Norm: $meanGradientsNorm%12.4f"
            }
          }
        }
        if (log)
          TrainingLogger.logger.info(message)
        writeSummary(lastStep, summaryTag, meanPerplexity)
        totalGradientsNorm = 0.0f
        totalLoss = 0.0f
        totalSrcWordCount = 0L
        totalTgtWordCount = 0L
      }
    }
  }
}

object TrainingLogger {
  private[TrainingLogger] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Training Logger"))
}
