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

import java.util.concurrent.atomic.AtomicLong

import scala.collection.mutable

/** Word counter data structure that uses a TRIE as the underlying data structure.
  *
  * @author Emmanouil Antonios Platanios
  */
case class TrieWordCounter() {
  protected val rootNode: TrieWordCounter.TrieNode = TrieWordCounter.TrieNode()

  protected var _totalCount: Long = {
    0L
  }

  def insertWord(word: String): Long = {
    _totalCount += 1
    var currentNode = rootNode
    for (char <- word)
      currentNode = currentNode.child(char)
    currentNode.incrementCount()
  }

  def insertWordWithCount(word: String, count: Long): Unit = {
    _totalCount += count
    var currentNode = rootNode
    for (char <- word)
      currentNode = currentNode.child(char)
    currentNode.setCount(count)
  }

  def apply(word: String): Long = {
    var currentNode = rootNode
    for (char <- word)
      currentNode = currentNode.child(char)
    currentNode.count
  }

  def totalCount: Long = {
    _totalCount
  }

  def words(sizeThreshold: Int = -1, countThreshold: Int = -1): Iterable[(Long, String)] = {
    if (sizeThreshold == -1 && countThreshold == -1) {
      rootNode.words.filter(_._2 != "")
    } else if (sizeThreshold == -1) {
      rootNode.words.filter(w => w._2 != "" && w._1 >= countThreshold)
    } else {
      val words = BoundedPriorityQueue[(Long, String)](sizeThreshold)
      rootNode.words.filter(_._2 != "").foreach {
        case (count, word) if countThreshold < 0 || count >= countThreshold => words += ((count, word))
        case _ => ()
      }
      words
    }
  }
}

object TrieWordCounter {
  case class TrieNode() {
    protected val _count   : AtomicLong                = new AtomicLong(0L)
    protected val _children: mutable.LongMap[TrieNode] = mutable.LongMap.empty[TrieNode]

    def count: Long = _count.get()
    def incrementCount(): Long = _count.incrementAndGet()
    def setCount(count: Long): Unit = _count.set(count)

    def child(char: Long): TrieNode = _children.getOrElseUpdate(char, TrieNode())
    def children: Seq[(Long, TrieNode)] = _children.toSeq

    def words: Iterable[(Long, String)] = {
      val words = (count, "") +: children.flatMap {
        case (char, childNode) =>
          childNode.words.map {
            case (count, word) => (count, char.toChar + word)
          }
      }
      words.filter(_._1 > 0)
    }
  }
}
