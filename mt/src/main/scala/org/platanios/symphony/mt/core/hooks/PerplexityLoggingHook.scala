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
import org.platanios.tensorflow.api.learn.{Counter, SessionCreator}
import org.platanios.tensorflow.api.learn.hooks._
import org.platanios.tensorflow.api.ops.variables.Variable

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

/** Hooks that logs the perplexity value.
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
class PerplexityLoggingHook(
    trigger: HookTrigger = StepHookTrigger(1),
    triggerAtEnd: Boolean = true,
    formatter: (Double, Long, Float) => String = (time, step, perplexity) => {
      f"($time%8.3f s) Step: $step%6d, Perplexity: $perplexity%.4f"
    }
) extends ModelDependentHook[
    (Output, Output),
    ((Tensor, Tensor), (Tensor, Tensor, Tensor)), ((Output, Output), (Output, Output, Output)),
    ((DataType, DataType), (DataType, DataType, DataType)), ((Shape, Shape), (Shape, Shape, Shape))] {
  private[this] var step: Variable = _
  private[this] var perplexity: Output   = _

  private[this] val internalTrigger: HookTrigger = trigger.copy()
  private[this] var lastStep       : Long        = 0L
  private[this] var shouldTrigger  : Boolean     = false

  override def begin(sessionCreator: SessionCreator): Unit = {
    step = Counter.get(Graph.Keys.GLOBAL_STEP, local = false).getOrElse(throw new IllegalStateException(
      s"A ${Graph.Keys.GLOBAL_STEP.name} variable should be created in order to use the 'PerplexityLoggingHook'."))
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
    if (shouldTrigger)
      Some(Hook.SessionRunArgs(fetches = Seq(step.value, perplexity)))
    else
      Some(Hook.SessionRunArgs(fetches = Seq(step.value)))
  }

  override def afterSessionRun[F, E, R](
      runContext: Hook.SessionRunContext[F, E, R],
      runResult: Hook.SessionRunResult[Seq[Output], Seq[Tensor]]
  )(implicit
      executableEv: Executable[E],
      fetchableEv: Fetchable.Aux[F, R]
  ): Unit = {
    lastStep = runResult.values.head.scalar.asInstanceOf[Long]
    if (shouldTrigger) {
      val perplexity = runResult.values(1).scalar.asInstanceOf[Float]
      val log = internalTrigger.updateLastTrigger(lastStep.toInt - 1).map(_._1) match {
        case Some(s) => formatter(s, lastStep, perplexity)
        case None => formatter(0.0, lastStep, perplexity)
      }
      PerplexityLoggingHook.logger.info(log)
    }
  }
}

object PerplexityLoggingHook {
  private[PerplexityLoggingHook] val logger = Logger(LoggerFactory.getLogger("Learn / Hooks / Perplexity Logging"))

  def apply(
      trigger: HookTrigger = StepHookTrigger(1),
      triggerAtEnd: Boolean = true,
      formatter: (Double, Long, Float) => String = { (time, step, perplexity) =>
        f"($time%.3f s) Step: $step%06d, Perplexity: $perplexity%.4f"
      }
  ): PerplexityLoggingHook = {
    new PerplexityLoggingHook(trigger, triggerAtEnd, formatter)
  }
}
