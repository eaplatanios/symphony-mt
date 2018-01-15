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

package org.platanios.symphony.mt.metrics

import org.junit.Test
import org.scalatest.junit.JUnitSuite

/** Contains tests for the machine translation metrics utilities.
  *
  * @author Emmanouil Antonios Platanios
  */
class UtilitiesSuite extends JUnitSuite {
  @Test def testCountNGrams(): Unit = {
    val sequence = Seq("this", "is", "a", "test", "and", "it", "is", "a", "great", "test", "what", "is", "this", "test")
    val counts = Utilities.countNGrams(sequence, 3)
    val expectedCounts = Map(
      Seq("this") -> 2L, Seq("is") -> 3L, Seq("a") -> 2L, Seq("test") -> 3L, Seq("and") -> 1L, Seq("it") -> 1L,
      Seq("great") -> 1L, Seq("what") -> 1L, Seq("this", "is") -> 1L, Seq("is", "a") -> 2L, Seq("a", "test") -> 1L,
      Seq("test", "and") -> 1L, Seq("and", "it") -> 1L, Seq("it", "is") -> 1L, Seq("a", "great") -> 1L,
      Seq("great", "test") -> 1L, Seq("test", "what") -> 1L, Seq("what", "is") -> 1L, Seq("is", "this") -> 1L,
      Seq("this", "test") -> 1L, Seq("this", "is", "a") -> 1L, Seq("is", "a", "test") -> 1L,
      Seq("a", "test", "and") -> 1L, Seq("test", "and", "it") -> 1L, Seq("and", "it", "is") -> 1L,
      Seq("it", "is", "a") -> 1L, Seq("is", "a", "great") -> 1L, Seq("a", "great", "test") -> 1L,
      Seq("great", "test", "what") -> 1L, Seq("test", "what", "is") -> 1L, Seq("what", "is", "this") -> 1L,
      Seq("is", "this", "test") -> 1L)
    assert(counts === expectedCounts)
  }
}
