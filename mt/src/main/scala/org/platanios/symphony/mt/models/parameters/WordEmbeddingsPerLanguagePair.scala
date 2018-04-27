///* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
// *
// * Licensed under the Apache License, Version 2.0 (the "License"); you may not
// * use this file except in compliance with the License. You may obtain a copy of
// * the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// * License for the specific language governing permissions and limitations under
// * the License.
// */
//
//package org.platanios.symphony.mt.models.parameters
//
//import org.platanios.symphony.mt.Language
//import org.platanios.symphony.mt.vocabulary.Vocabulary
//import org.platanios.tensorflow.api._
//
///**
//  * @author Emmanouil Antonios Platanios
//  */
//class WordEmbeddingsPerLanguagePair protected (
//    override val embeddingsSize: Int,
//    override val mergedEmbeddings: Boolean = false,
//    override val mergedProjections: Boolean = false
//) extends WordEmbeddingsPerLanguage(embeddingsSize, mergedEmbeddings, mergedProjections) {
//  override def createWordEmbeddings(languages: Seq[(Language, Vocabulary)]): Seq[Output] = {
//    val languagePairs = languages.combinations(2).map(c => (c(0), c(1))).toSeq
//    val embeddingsInitializer = tf.RandomUniformInitializer(-0.1f, 0.1f)
//    if (!mergedEmbeddings) {
//      languagePairs.flatMap(pair => tf.variableScope(s"${pair._1._1.abbreviation}-${pair._2._1.abbreviation}") {
//        Seq(
//          tf.variable(pair._1._1.name, FLOAT32, Shape(pair._1._2.size, embeddingsSize), embeddingsInitializer).value,
//          tf.variable(pair._2._1.name, FLOAT32, Shape(pair._2._2.size, embeddingsSize), embeddingsInitializer).value)
//      })
//    } else {
//      val vocabSizes = languagePairs.map(pair => Seq(pair._1._2.size, pair._2._2.size))
//      val vocabPairSizes = vocabSizes.map(p => p(0) + p(1))
//      val merged = tf.variable(
//        "Embeddings", FLOAT32, Shape(vocabPairSizes.sum, embeddingsSize), embeddingsInitializer).value
//
//      val sizes = tf.createWithNameScope("VocabularySizes")(tf.stack(vocabSizes.map(tf.constant(_))))
//      val offsets = tf.concatenate(Seq(tf.zeros(sizes.dataType, Shape(1)), tf.cumsum(sizes)(0 :: -1)))
//      Seq(merged, offsets)
//    }
//  }
//
//  override def embeddingLookup(
//      embeddingTables: Seq[Output],
//      languageIds: Seq[Output],
//      languageId: Output,
//      keys: Output,
//      context: Option[(Output, Output)]
//  ): Output = {
//    if (!mergedEmbeddings) {
//      val predicates = embeddingTables.zip(languageIds).map {
//        case (embeddings, langId) => (tf.equal(languageId, langId), () => embeddings)
//      }
//      val default = () => embeddingTables.head
//      tf.cases(predicates, default).gather(keys)
//    } else {
//      val merged = embeddingTables(0)
//      val offsets = embeddingTables(1)
//      merged.gather(keys + offsets.gather(languageId))
//    }
//  }
//}
//
//object WordEmbeddingsPerLanguagePair {
//  def apply(
//      embeddingsSize: Int,
//      mergedEmbeddings: Boolean = false,
//      mergedProjections: Boolean = false
//  ): WordEmbeddingsPerLanguagePair = {
//    new WordEmbeddingsPerLanguagePair(embeddingsSize, mergedEmbeddings, mergedProjections)
//  }
//}
