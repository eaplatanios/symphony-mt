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

  def layer: Layer[(Output, Output), Tuple[Output, Seq[S]]]
}

class GNMTEncoder[S, SS](
    val configuration: Configuration[S, SS],
    override val name: String = "GNMTEncoder"
)(implicit
  evS: WhileLoopVariable.Aux[S, SS],
    evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
) extends Encoder[S, SS] {
  def timeMajor: Boolean = configuration.dataTimeMajor

  def layer: Layer[(Output, Output), Tuple[Output, Seq[S]]] = {
    new Layer[(Output, Output), Tuple[Output, Seq[S]]](name) {
      override val layerType: String = "GNMTEncoder"

      override def forward(
          input: (Output, Output),
          mode: Mode
      ): LayerInstance[(Output, Output), Tuple[Output, Seq[S]]] = {
        val dataType = configuration.dataType
        tf.createWithNameScope(uniquifiedName) {
          // Keep track of trainable and non-trainable variables
          var trainableVariables = Set.empty[Variable]
          var nonTrainableVariables = Set.empty[Variable]

          // Embeddings
          val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
          val embeddings = variable(
            "Embeddings", dataType, Shape(configuration.srcVocabSize, configuration.numUnits), embeddingsInitializer)
          val embeddedInput = tf.embeddingLookup(embeddings, input._1)
          trainableVariables += embeddings

          // Bidirectional Layers
          val (output, state) = {
            if (configuration.numBiLayers > 0) {
              val biCellFw = GNMTModel.multiCell(
                configuration.cell, configuration.numUnits, dataType, configuration.numBiLayers,
                configuration.numBiResLayers, configuration.dropout, configuration.residualFn, 0, configuration.numGPUs,
                configuration.randomSeed, s"$name/MultiBiCellFw")
              val biCellBw = GNMTModel.multiCell(
                configuration.cell, configuration.numUnits, dataType, configuration.numBiLayers,
                configuration.numBiResLayers, configuration.dropout, configuration.residualFn, configuration.numBiLayers,
                configuration.numGPUs, configuration.randomSeed, s"$name/MultiBiCellBw")
              val biCellInstanceFw = biCellFw.createCell(mode, embeddedInput.shape)
              val biCellInstanceBw = biCellBw.createCell(mode, embeddedInput.shape)
              val unmergedBiTuple = tf.bidirectionalDynamicRNN(
                biCellInstanceFw.cell, biCellInstanceBw.cell, embeddedInput, null, null, timeMajor,
                configuration.parallelIterations, configuration.swapMemory, input._2,
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
          val uniCell = GNMTModel.multiCell(
            configuration.cell, configuration.numUnits, dataType, configuration.numUniLayers,
            configuration.numUniResLayers, configuration.dropout, configuration.residualFn,
            2 * configuration.numBiLayers, configuration.numGPUs, configuration.randomSeed, s"$name/MultiUniCell")
          val uniCellInstance = uniCell.createCell(mode, output.shape)
          val uniTuple = tf.dynamicRNN(
            uniCellInstance.cell, output, null, timeMajor, configuration.parallelIterations,
            configuration.swapMemory, input._2, s"$uniquifiedName/UniDirectionalLayers")
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
  def apply[S, SS](configuration: Configuration[S, SS], name: String = "GNMTEncoder")(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evSDropout: ops.rnn.cell.DropoutWrapper.Supported[S]
  ): Encoder[S, SS] = {
    new GNMTEncoder(configuration, name)(evS, evSDropout)
  }
}
