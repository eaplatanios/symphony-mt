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

package org.platanios.symphony.mt.metrics

import org.junit.Test
import org.scalatest.Matchers
import org.scalatest.junit.JUnitSuite

/** Contains tests for the the BLEU score calculation.
  *
  * @author Emmanouil Antonios Platanios
  */
class BLEUSuite extends JUnitSuite with Matchers {
  @Test def testbleuSingleReference(): Unit = {
    val reference = Seq(
      Seq(Seq("a", "b", "c", "d", "e")),
      Seq(Seq("a", "b", "c", "d", "e")),
      Seq(Seq("a", "b", "c", "d", "e")),
      Seq(Seq("a", "b", "c", "d", "e")))
    val hypothesis1 = Seq(
      Seq("a", "b", "c", "d", "e"),
      Seq("a", "b", "c", "d", "e", "f"),
      Seq("a", "c", "d", "e"),
      Seq("a", "b", "x", "d", "e"))
    val hypothesis2 = Seq(
      Seq("a", "b", "c", "d", "e"),
      Seq("a", "b", "c", "d", "e", "f", "g"),
      Seq("a", "b", "c", "d", "e"),
      Seq("a", "b", "x", "d", "e"))
    assert(BLEU.bleu(reference, hypothesis1, maxOrder = 4, smooth = false).score === 0.666112590882706 +- 1e-14)
    assert(BLEU.bleu(reference, hypothesis1, maxOrder = 5, smooth = false).score === 0.628973162582494 +- 1e-14)
    assert(BLEU.bleu(reference, hypothesis1, maxOrder = 4, smooth = true).score === 0.697390001644341 +- 1e-14)
    assert(BLEU.bleu(reference, hypothesis1, maxOrder = 5, smooth = true).score === 0.676722822649727 +- 1e-14)
    assert(BLEU.bleu(reference, hypothesis2, maxOrder = 4, smooth = false).score === 0.713449180942819 +- 1e-14)
    assert(BLEU.bleu(reference, hypothesis2, maxOrder = 5, smooth = false).score === 0.664483858835372 +- 1e-14)
    assert(BLEU.bleu(reference, hypothesis2, maxOrder = 4, smooth = true).score === 0.734621235774005 +- 1e-14)
    assert(BLEU.bleu(reference, hypothesis2, maxOrder = 5, smooth = true).score === 0.698623472351266 +- 1e-14)
  }

  @Test def testbleuMultipleReferences(): Unit = {
    val reference = Seq(
      Seq(Seq("a", "b", "c", "d", "e"), Seq("a", "c", "d", "e")),
      Seq(Seq("a", "b", "c", "d", "e")),
      Seq(Seq("a", "b", "c", "d", "e"), Seq("a", "c", "d", "e")),
      Seq(Seq("a", "b", "c", "d", "e")))
    val hypothesis1 = Seq(
      Seq("a", "b", "c", "d", "e"),
      Seq("a", "b", "c", "d", "e", "f"),
      Seq("a", "c", "d", "e"),
      Seq("a", "b", "x", "d", "e"))
    val hypothesis2 = Seq(
      Seq("a", "b", "c", "d", "e"),
      Seq("a", "b", "c", "d", "e", "f", "g"),
      Seq("a", "b", "c", "d", "e"),
      Seq("a", "b", "x", "d", "e"))
    assert(BLEU.bleu(reference, hypothesis1, maxOrder = 4, smooth = false).score === 0.742956966550210 +- 1e-14)
    assert(BLEU.bleu(reference, hypothesis1, maxOrder = 5, smooth = false).score === 0.686380486706770 +- 1e-14)
    assert(BLEU.bleu(reference, hypothesis1, maxOrder = 4, smooth = true).score === 0.765782309941811 +- 1e-14)
    assert(BLEU.bleu(reference, hypothesis1, maxOrder = 5, smooth = true).score === 0.729313938509302 +- 1e-14)
    assert(BLEU.bleu(reference, hypothesis2, maxOrder = 4, smooth = false).score === 0.713449180942819 +- 1e-14)
    assert(BLEU.bleu(reference, hypothesis2, maxOrder = 5, smooth = false).score === 0.664483858835372 +- 1e-14)
    assert(BLEU.bleu(reference, hypothesis2, maxOrder = 4, smooth = true).score === 0.734621235774005 +- 1e-14)
    assert(BLEU.bleu(reference, hypothesis2, maxOrder = 5, smooth = true).score === 0.698623472351266 +- 1e-14)
  }
}
