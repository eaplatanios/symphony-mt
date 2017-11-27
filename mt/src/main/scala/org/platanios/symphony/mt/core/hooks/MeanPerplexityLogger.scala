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
import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.core.client.{Executable, Fetchable, Session}
import org.platanios.tensorflow.api.learn.hooks._
import org.platanios.tensorflow.api.learn.{Counter, SessionCreator}
import org.platanios.tensorflow.api.ops.variables.Variable

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

/** Hooks that logs the mean perplexity value across multiple steps.
  *
  * Whenever this hook is triggered it logs the mean perplexity over all steps since the last time it was triggered.
  *
  * @param  trigger      Hook trigger specifying when this hook is triggered (i.e., when it executes). If you only want
  *                      to log the tensor values at the end of a run and not during, then you should set `trigger` to
  *                      [[NoHookTrigger]] and `logAtEnd` to `true`.
  * @param  triggerAtEnd If `true`, this hook will be triggered at the end of the run. Note that if this flag is set to
  *                      `true`, then `tensors` must be computable without using a feed map for the [[Session.run()]]
  *                      call.
  * @param  formatter    Function used to format the message that is being logged. It takes the time taken since the
  *                      last logged message, the current step, and the current loss value, as input, and returns a
  *                      string to log.
  *
  * @author Emmanouil Antonios Platanios
  */
class MeanPerplexityLogger(
    trigger: HookTrigger = StepHookTrigger(1),
    triggerAtEnd: Boolean = true,
    formatter: (Double, Long, Float) => String = (time, step, perplexity) => {
      f"($time%8.3f s) Step: $step%6d, Mean Perplexity: $perplexity%.4f"
    }
) extends ModelDependentHook[
    (Output, Output),
    ((Tensor, Tensor), (Tensor, Tensor, Tensor)), ((Output, Output), (Output, Output, Output)),
    ((DataType, DataType), (DataType, DataType, DataType)), ((Shape, Shape), (Shape, Shape, Shape))] {
  private[this] var step      : Variable = _
  private[this] var perplexity: Output   = _

  private[this] val internalTrigger: HookTrigger = trigger.copy()
  private[this] var lastStep       : Long        = 0L
  private[this] var shouldTrigger  : Boolean     = false
  private[this] var totalPerplexity: Float       = 0.0f
  private[this] var totalSteps     : Long        = 0L

  override def begin(sessionCreator: SessionCreator): Unit = {
    step = Counter.get(Graph.Keys.GLOBAL_STEP, local = false).getOrElse(throw new IllegalStateException(
      s"A ${Graph.Keys.GLOBAL_STEP.name} variable should be created in order to use the 'MeanPerplexityLogger'."))
    internalTrigger.reset()
    shouldTrigger = false
    perplexity = modelInstance.loss.map(_.cast(FLOAT32)).map(l => {
      tf.exp(l * tf.size(modelInstance.output._2) / tf.sum(modelInstance.output._2))
    }).orNull
  }

  override def afterSessionCreation(session: Session): Unit = {
    lastStep = session.run(fetches = step.value).scalar.asInstanceOf[Long]
  }

  override def beforeSessionRun[F, E, R](runContext: Hook.SessionRunContext[F, E, R])(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Option[Hook.SessionRunArgs[Seq[Output], Traversable[Op], Seq[Tensor]]] = {
    shouldTrigger = perplexity != null && internalTrigger.shouldTriggerForStep(lastStep.toInt)
    Some(Hook.SessionRunArgs(fetches = Seq(step.value, perplexity)))
  }

  override def afterSessionRun[F, E, R](
      runContext: Hook.SessionRunContext[F, E, R],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor]]
  )(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Unit = {
    lastStep = runResult.values.head.scalar.asInstanceOf[Long]
    totalPerplexity += runResult.values(1).scalar.asInstanceOf[Float]
    totalSteps += 1L
    if (shouldTrigger) {
      val meanPerplexity = totalPerplexity / totalSteps
      val log = internalTrigger.updateLastTrigger(lastStep.toInt - 1).map(_._1) match {
        case Some(s) => formatter(s, lastStep, meanPerplexity)
        case None => formatter(0.0, lastStep, meanPerplexity)
      }
      MeanPerplexityLogger.logger.info(log)
      totalPerplexity = 0.0f
      totalSteps = 0L
    }
  }
}

object MeanPerplexityLogger {
  private[MeanPerplexityLogger] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Mean Perplexity Logger"))

  def apply(
      trigger: HookTrigger = StepHookTrigger(1),
      triggerAtEnd: Boolean = true,
      formatter: (Double, Long, Float) => String = { (time, step, perplexity) =>
        f"($time%8.3f s) Step: $step%6d, Mean Perplexity: $perplexity%.4f"
      }
  ): MeanPerplexityLogger = {
    new MeanPerplexityLogger(trigger, triggerAtEnd, formatter)
  }
}
