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

package org.platanios.symphony.mt.experiments

import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.evaluation
import org.platanios.symphony.mt.evaluation._
import org.platanios.symphony.mt.vocabulary.Vocabulary

/**
  * @author Emmanouil Antonios Platanios
  */
sealed trait Metric {
  val header: String
}

case object BLEU extends Metric {
  override val header: String = "BLEU"
}

case object Meteor extends Metric {
  override val header: String = "Meteor"
}

case object TER extends Metric {
  override val header: String = "TER"
}

case object HypLen extends Metric {
  override val header: String = "HypLen"
}

case object RefLen extends Metric {
  override val header: String = "RefLen"
}

case object SentenceCount extends Metric {
  override val header: String = "#Sentences"
}

object Metric {
  @throws[IllegalArgumentException]
  def fromHeader(header: String): Metric = {
    header match {
      case "BLEU" => BLEU
      case "Meteor" => Meteor
      case "TER" => TER
      case "HypLen" => HypLen
      case "RefLen" => RefLen
      case "#Sentences" => SentenceCount
      case _ => throw new IllegalArgumentException(s"'$header' does not represent a valid metric header name.")
    }
  }

  @throws[IllegalArgumentException]
  def cliToMTMetric(metric: String)(implicit languages: Seq[(Language, Vocabulary)]): MTMetric = {
    metric.split(":") match {
      case Array("bleu") => evaluation.BLEU()(languages)
      case Array("bleu", maxOrder) => evaluation.BLEU(maxOrder.toInt)(languages)
      case Array("bleu", maxOrder, smooth) => evaluation.BLEU(maxOrder.toInt, smooth.toBoolean)(languages)
      case Array("meteor") => evaluation.Meteor()(languages)
      case Array("ter") => evaluation.TER()(languages)
      case Array("hyp_len") => evaluation.SentenceLength(forHypothesis = true, name = "HypLen")
      case Array("ref_len") => evaluation.SentenceLength(forHypothesis = false, name = "RefLen")
      case Array("sen_cnt") => evaluation.SentenceCount(name = "#Sentences")
      case _ => throw new IllegalArgumentException(s"'$metric' does not represent a valid metric.")
    }
  }
}
