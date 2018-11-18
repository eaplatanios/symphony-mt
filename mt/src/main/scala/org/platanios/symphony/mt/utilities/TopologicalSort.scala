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

package org.platanios.symphony.mt.utilities

/**
  * @author Emmanouil Antonios Platanios
  */
object TopologicalSort {
  def sort[T](values: Set[T], requirements: T => Set[T]): Option[Seq[T]] = {
    // Collect all values by following the dependencies formed by their requirements.
    var allValues = values
    var continue = true
    while (continue) {
      val previousSize = allValues.size
      allValues ++= allValues.flatMap(requirements)
      continue = allValues.size != previousSize
    }

    // Create graph edges and nodes.
    val edges = allValues.flatMap(v => requirements(v).map(r => (v, r)))
    val nodes = allValues.map(value => {
      value -> Node(value, edges.filter(e => e._2 == value).map(_._1))
    }).toMap

    // Create initial set of unmarked nodes, which consists of all nodes in the graph.
    var unmarkedNodes = nodes.values.toSet

    var sortedValues = Seq.empty[T]
    var hasCycle = false

    def visit(node: Node[T]): Unit = {
      if (!node.isPermanentlyMarked && !node.isTemporarilyMarked) {
        node.isTemporarilyMarked = true
        node.dependants.foreach(d => visit(nodes(d)))
        node.isTemporarilyMarked = false
        node.isPermanentlyMarked = true
        unmarkedNodes -= node
        sortedValues = node.value +: sortedValues
      } else if (node.isTemporarilyMarked) {
        hasCycle = true
      }
    }

    // Perform depth-first search.
    while (!hasCycle && unmarkedNodes.nonEmpty) {
      visit(unmarkedNodes.head)
    }

    if (hasCycle)
      None
    else
      Some(sortedValues)
  }

  private[TopologicalSort] case class Node[T](value: T, dependants: Set[T]) {
    private[TopologicalSort] var isTemporarilyMarked: Boolean = false
    private[TopologicalSort] var isPermanentlyMarked: Boolean = false
  }
}
