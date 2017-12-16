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

package org.platanios.symphony.mt.models.rnn

import org.platanios.symphony.mt.core.Environment
import org.platanios.symphony.mt.data.{DataConfig, Vocabulary}
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.{Layer, LayerInstance}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.rnn.cell.Tuple

import scala.language.postfixOps

/**
  * @author Emmanouil Antonios Platanios
  */
trait Encoder[S, SS] {
  val name: String = "Encoder"

  val env       : Environment
  val dataConfig: DataConfig

  def layer: Layer[(Output, Output), Tuple[Output, Seq[S]]]
}

class GNMTEncoder[S, SS](
    override val env: Environment,
    val config: Configuration[S, SS],
    override val dataConfig: DataConfig,
    val srcVocabulary: Vocabulary,
    override val name: String = "GNMTEncoder"
)(implicit
  evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) extends Encoder[S, SS] {
  def timeMajor: Boolean = dataConfig.timeMajor

  def layer: Layer[(Output, Output), Tuple[Output, Seq[S]]] = {
    new Layer[(Output, Output), Tuple[Output, Seq[S]]](name) {
      override val layerType: String = "GNMTEncoder"

      override def forward(
          input: (Output, Output),
          mode: Mode
      ): LayerInstance[(Output, Output), Tuple[Output, Seq[S]]] = {
        val dataType = config.dataType
        tf.createWithNameScope(uniquifiedName) {
          // Keep track of trainable and non-trainable variables
          var trainableVariables = Set.empty[Variable]
          var nonTrainableVariables = Set.empty[Variable]

          // Embeddings
          val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
          val embeddings = variable(
            "Embeddings", dataType, Shape(srcVocabulary.size, config.numUnits), embeddingsInitializer)
          val embeddedInput = tf.embeddingLookup(embeddings, input._1)
          trainableVariables += embeddings

          // Bidirectional Layers
          val (output, state) = {
            if (config.numBiLayers > 0) {
              val biCellFw = Model.multiCell(
                config.cell, config.numUnits, dataType, config.numBiLayers,
                config.numBiResLayers, config.dropout, config.residualFn, 0, env.numGPUs,
                env.randomSeed, s"$name/MultiBiCellFw")
              val biCellBw = Model.multiCell(
                config.cell, config.numUnits, dataType, config.numBiLayers,
                config.numBiResLayers, config.dropout, config.residualFn, config.numBiLayers,
                env.numGPUs, env.randomSeed, s"$name/MultiBiCellBw")
              val biCellInstanceFw = biCellFw.createCell(mode, embeddedInput.shape)
              val biCellInstanceBw = biCellBw.createCell(mode, embeddedInput.shape)
              val unmergedBiTuple = tf.bidirectionalDynamicRNN(
                biCellInstanceFw.cell, biCellInstanceBw.cell, embeddedInput, null, null, timeMajor,
                env.parallelIterations, env.swapMemory, input._2,
                s"$uniquifiedName/BiDirectionalLayers")
              val mergedBiTuple = Tuple(
                tf.concatenate(Seq(unmergedBiTuple._1.output, unmergedBiTuple._2.output), -1), unmergedBiTuple._2.state)
              trainableVariables ++= biCellInstanceFw.trainableVariables ++ biCellInstanceBw.trainableVariables
              nonTrainableVariables ++= biCellInstanceFw.nonTrainableVariables ++ biCellInstanceBw.nonTrainableVariables
              (mergedBiTuple.output, mergedBiTuple.state)
            } else {
              (embeddedInput, Seq.empty[S])
            }
          }

          // Unidirectional Layers
          val uniCell = Model.multiCell(
            config.cell, config.numUnits, dataType, config.numUniLayers,
            config.numUniResLayers, config.dropout, config.residualFn,
            2 * config.numBiLayers, env.numGPUs, env.randomSeed, s"$name/MultiUniCell")
          val uniCellInstance = uniCell.createCell(mode, output.shape)
          val uniTuple = tf.dynamicRNN(
            uniCellInstance.cell, output, null, timeMajor, env.parallelIterations,
            env.swapMemory, input._2, s"$uniquifiedName/UniDirectionalLayers")
          trainableVariables ++= uniCellInstance.trainableVariables
          nonTrainableVariables ++= uniCellInstance.nonTrainableVariables

          // Pass all of the encoder's state except for the first bi-directional layer's state, to the decoder.
          val tuple = Tuple(uniTuple.output, state ++ uniTuple.state)
          LayerInstance(input, tuple, trainableVariables, nonTrainableVariables)
        }
      }
    }
  }
}

object GNMTEncoder {
  def apply[S, SS](
      env: Environment,
      config: Configuration[S, SS],
      dataConfig: DataConfig,
      srcVocabulary: Vocabulary,
      name: String = "GNMTEncoder"
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): Encoder[S, SS] = {
    new GNMTEncoder(env, config, dataConfig, srcVocabulary, name)(evS, evSDropout)
  }
}
