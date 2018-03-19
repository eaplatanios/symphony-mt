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

package org.platanios.symphony.mt.data.processors

import better.files._
import com.typesafe.scalalogging.Logger
import org.platanios.symphony.mt.Language
import org.platanios.symphony.mt.Language._
import org.platanios.symphony.mt.data.{newReader, newWriter}
import org.slf4j.LoggerFactory

import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
trait Tokenizer {
  def tokenizedFile(originalFile: File): File = {
    val fileName = originalFile.nameWithoutExtension(includeAll = false) + s".tok${originalFile.extension().get}"
    originalFile.sibling(fileName)
  }

  def tokenize(sentence: String, language: Language): String

  def tokenizeCorpus(file: File, language: Language, bufferSize: Int = 8192): File = {
    val tokenized = tokenizedFile(file)
    if (tokenized.notExists) {
      Tokenizer.logger.info(s"Tokenizing '$file'.")
      val tokenizedWriter = newWriter(tokenized)
      newReader(file).lines().toAutoClosedIterator.foreach(sentence => {
        tokenizedWriter.write(s"${tokenize(sentence, language)}\n")
      })
      tokenizedWriter.flush()
      tokenizedWriter.close()
      Tokenizer.logger.info(s"Created tokenized file '$tokenized'.")
    }
    tokenized
  }
}

object Tokenizer {
  private[data] val logger = Logger(LoggerFactory.getLogger("Data / Tokenizer"))
}

object NoTokenizer extends Tokenizer {
  override def tokenizedFile(originalFile: File): File = originalFile
  override def tokenize(sentence: String, language: Language): String = sentence
  override def tokenizeCorpus(file: File, language: Language, bufferSize: Int = 8192): File = file
}

/** Tokenizer used by the Moses library.
  *
  * The code in this class is a Scala translation of the
  * [original Moses code](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl).
  *
  * @param  aggressiveHyphenSplitting If `true`, hyphens will be aggressively split and replaced by `"@-@"` tokens.
  * @param  escapeSpecialCharacters   If `true`, various special characters (such as a double quote, for example) will
  *                                   be escaped.
  * @param  nonBreakingPrefixes       Optional map containing non-breaking prefixes for each language. These are
  *                                   prefixes that are not at the end of sentences (i.e., sentence-breaking).
  */
case class MosesTokenizer(
    aggressiveHyphenSplitting: Boolean = false,
    escapeSpecialCharacters: Boolean = true,
    nonBreakingPrefixes: Map[Language, MosesTokenizer.NonBreakingPrefixes] = MosesTokenizer.defaultNonBreakingPrefixes
) extends Tokenizer {
  private val ignoredRegex                  : Regex = """[\000-\037]""".r
  private val whitespaceRegex               : Regex = """\s+""".r
  private val periodRegex                   : Regex = """\.""".r
  private val alphabeticRegex               : Regex = """\p{IsAlpha}""".r
  private val notLowercaseRegex             : Regex = """^[\p{IsLower}]""".r
  private val notDigitsRegex                : Regex = """^[0-9]+""".r
  private val symbolsRegex                  : Regex = """([^\p{IsAlnum}\s\.\'\`\,\-])""".r
  private val fiSvColonRegex1               : Regex = """([^\p{IsAlnum}\s\.\:\'\`\,\-])""".r
  private val fiSvColonRegex2               : Regex = """(:)(?=$|[^\p{IsLl}])""".r
  private val aggressiveHyphenSplittingRegex: Regex = """([\p{IsAlnum}])\-(?=[\p{IsAlnum}])""".r
  private val multiDotRegex1                : Regex = """\.([\.]+)""".r
  private val multiDotRegex2                : Regex = """DOTMULTI\.""".r
  private val multiDotRegex3                : Regex = """DOTMULTI\.([^\.])""".r
  private val multiDotRegex4                : Regex = """DOTDOTMULTI""".r
  private val multiDotRegex5                : Regex = """DOTMULTI""".r
  private val commaRegex1                   : Regex = """([^\p{IsDigit}])[,]""".r
  private val commaRegex2                   : Regex = """[,]([^\p{IsDigit}])""".r
  private val enContractionsRegex1          : Regex = """([^\p{IsAlpha}])[']([^\p{IsAlpha}])""".r
  private val enContractionsRegex2          : Regex = """([^\p{IsAlpha}\p{IsDigit}])[']([\p{IsAlpha}])""".r
  private val enContractionsRegex3          : Regex = """([\p{IsAlpha}])[']([^\p{IsAlpha}])""".r
  private val enContractionsRegex4          : Regex = """([\p{IsAlpha}])[']([\p{IsAlpha}])""".r
  private val enContractionsRegex5          : Regex = """([\p{IsDigit}])[']([s])""".r
  private val frItGaContractionsRegex1      : Regex = """([^\p{IsAlpha}])[']([^\p{IsAlpha}])""".r
  private val frItGaContractionsRegex2      : Regex = """([^\p{IsAlpha}])[']([\p{IsAlpha}])""".r
  private val frItGaContractionsRegex3      : Regex = """([\p{IsAlpha}])[']([^\p{IsAlpha}])""".r
  private val frItGaContractionsRegex4      : Regex = """([\p{IsAlpha}])[']([\p{IsAlpha}])""".r
  private val otherContractionsRegex        : Regex = """\'""".r
  private val wordRegex                     : Regex = """^(\S+)\.$""".r
  private val endOfSentenceRegex            : Regex = """\.\' ?$""".r
  private val escapeRegex1                  : Regex = """\&""".r
  private val escapeRegex2                  : Regex = """\|""".r
  private val escapeRegex3                  : Regex = """\<""".r
  private val escapeRegex4                  : Regex = """\>""".r
  private val escapeRegex5                  : Regex = """\'""".r
  private val escapeRegex6                  : Regex = """\"""".r
  private val escapeRegex7                  : Regex = """\[""".r
  private val escapeRegex8                  : Regex = """\]""".r

  override def tokenize(sentence: String, language: Language): String = {
    var tokenized = sentence.trim

    // Remove ASCII junk characters.
    tokenized = whitespaceRegex.replaceAllIn(tokenized, " ")
    tokenized = ignoredRegex.replaceAllIn(tokenized, "")

    // Separate out all other special characters.
    if (language == finnish || language == swedish) {
      // In Finnish and Swedish, the colon can be used inside words as an apostrophe-like character. For example:
      // "USA:n", "20:een", "EU:ssa", "USA:s", "S:t", etc.
      tokenized = fiSvColonRegex1.replaceAllIn(tokenized, " $1 ")
      // If a colon is not immediately followed by lower-case characters, separate it out anyway.
      tokenized = fiSvColonRegex2.replaceAllIn(tokenized, " $1 ")
    } else {
      tokenized = symbolsRegex.replaceAllIn(tokenized, " $1 ")
    }

    // Aggressive hyphen splitting.
    if (aggressiveHyphenSplitting)
      tokenized = aggressiveHyphenSplittingRegex.replaceAllIn(tokenized, "$1 @-@ ")

    // Multi-dots stay together.
    tokenized = multiDotRegex1.replaceAllIn(tokenized, " DOTMULTI$1")
    while (multiDotRegex2.findFirstIn(tokenized).isDefined) {
      tokenized = multiDotRegex3.replaceAllIn(tokenized, "DOTDOTMULTI $1")
      tokenized = multiDotRegex2.replaceAllIn(tokenized, "DOTDOTMULTI")
    }

    // Separate out "," except if within numbers, and after a number if it's at the end of a sentence.
    tokenized = commaRegex1.replaceAllIn(tokenized, "$1 , ")
    tokenized = commaRegex2.replaceAllIn(tokenized, " , $1")

    // Handle contractions.
    if (language == english) {
      // Split contractions right.
      tokenized = enContractionsRegex1.replaceAllIn(tokenized, "$1 ' $2")
      tokenized = enContractionsRegex2.replaceAllIn(tokenized, "$1 ' $2")
      tokenized = enContractionsRegex3.replaceAllIn(tokenized, "$1 ' $2")
      tokenized = enContractionsRegex4.replaceAllIn(tokenized, "$1 '$2")
      // Special case for years, like "1990's".
      tokenized = enContractionsRegex5.replaceAllIn(tokenized, "$1 '$2")
    } else if (language == french || language == italian || language == irish) {
      // Split contractions left.
      tokenized = frItGaContractionsRegex1.replaceAllIn(tokenized, "$1 ' $2")
      tokenized = frItGaContractionsRegex2.replaceAllIn(tokenized, "$1 ' $2")
      tokenized = frItGaContractionsRegex3.replaceAllIn(tokenized, "$1 ' $2")
      tokenized = frItGaContractionsRegex4.replaceAllIn(tokenized, "$1' $2")
    } else {
      tokenized = otherContractionsRegex.replaceAllIn(tokenized, " ' ")
    }

    // Tokenize words.
    val words = whitespaceRegex.split(tokenized)
    tokenized = words.zipWithIndex.map(word => {
      val wordMatch = wordRegex.findFirstMatchIn(word._1)
      if (wordMatch.isEmpty) {
        word._1
      } else {
        val prefix = wordMatch.get.group(1)
        if (periodRegex.findFirstIn(prefix).isDefined && alphabeticRegex.findFirstIn(prefix).isDefined ||
            nonBreakingPrefixes(language).prefixes.contains(prefix) ||
            word._2 < words.length - 1 && notLowercaseRegex.findFirstIn(words(word._2 + 1)).isDefined ||
            (word._2 < words.length - 1 &&
                notDigitsRegex.findFirstIn(words(word._2 + 1)).isDefined &&
                nonBreakingPrefixes(language).numericPrefixes.contains(prefix))) {
          word._1
        } else {
          s"$prefix ."
        }
      }
    }).mkString(" ")

    // Clean up extraneous spaces.
    tokenized = tokenized.trim
    tokenized = whitespaceRegex.replaceAllIn(tokenized, " ")

    // Fix ".'" at the end of the sentence.
    tokenized = endOfSentenceRegex.replaceFirstIn(tokenized, " . ' ")

    // Restore multi-dots.
    while (multiDotRegex4.findFirstIn(tokenized).isDefined)
      tokenized = multiDotRegex4.replaceAllIn(tokenized, "DOTMULTI.")
    tokenized = multiDotRegex5.replaceAllIn(tokenized, ".")

    // Escape special characters.
    if (escapeSpecialCharacters) {
      tokenized = escapeRegex1.replaceAllIn(tokenized, "&amp;")
      tokenized = escapeRegex2.replaceAllIn(tokenized, "&#124;")
      tokenized = escapeRegex3.replaceAllIn(tokenized, "&lt;")
      tokenized = escapeRegex4.replaceAllIn(tokenized, "&gt;")
      tokenized = escapeRegex5.replaceAllIn(tokenized, "&apos;")
      tokenized = escapeRegex6.replaceAllIn(tokenized, "&quot;")
      tokenized = escapeRegex7.replaceAllIn(tokenized, "&#91;")
      tokenized = escapeRegex8.replaceAllIn(tokenized, "&#93;")
    }

    tokenized
  }
}

object MosesTokenizer {
  case class NonBreakingPrefixes(prefixes: Set[String] = Set.empty, numericPrefixes: Set[String] = Set.empty)

  private[data] val defaultNonBreakingPrefixes = Map(
    catalan -> NonBreakingPrefixes(
      prefixes = Set(
        "Dr", "Dra", "pàg", "p", "c", "av", "Sr", "Sra", "adm", "esq", "Prof", "S.A", "S.L", "p.e", "ptes", "Sta", "St",
        "pl", "màx", "cast", "dir", "nre", "fra", "admdora", "Emm", "Excma", "espf", "dc", "admdor", "tel", "angl",
        "aprox", "ca", "dept", "dj", "dl", "dt", "ds", "dg", "dv", "ed", "entl", "al", "i.e", "maj", "smin", "n", "núm",
        "pta", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
        "V", "W", "X", "Y", "Z")),
    chinese -> NonBreakingPrefixes(
      prefixes = Set(
        "A", "Ā", "B", "C", "Č", "D", "E", "Ē", "F", "G", "Ģ", "H", "I", "Ī", "J", "K", "Ķ", "L", "Ļ", "M", "N", "Ņ",
        "O", "P", "Q", "R", "S", "Š", "T", "U", "Ū", "V", "W", "X", "Y", "Z", "Ž"),
      numericPrefixes = Set("No", "Nr")),
    czech -> NonBreakingPrefixes(
      prefixes = Set(
        "Bc", "BcA", "Ing", "Ing.arch", "MUDr", "MVDr", "MgA", "Mgr", "JUDr", "PhDr", "RNDr", "PharmDr", "ThLic",
        "ThDr", "Ph.D", "Th.D", "prof", "doc", "CSc", "DrSc", "dr. h. c", "PaedDr", "Dr", "PhMr", "DiS", "abt", "ad",
        "a.i", "aj", "angl", "anon", "apod", "atd", "atp", "aut", "bd", "biogr", "b.m", "b.p", "b.r", "cca", "cit",
        "cizojaz", "c.k", "col", "čes", "čín", "čj", "ed", "facs", "fasc", "fol", "fot", "franc", "h.c", "hist", "hl",
        "hrsg", "ibid", "il", "ind", "inv.č", "jap", "jhdt", "jv", "koed", "kol", "korej", "kl", "krit", "lat", "lit",
        "m.a", "maď", "mj", "mp", "násl", "např", "nepubl", "něm", "no", "nr", "n.s", "okr", "odd", "odp", "obr", "opr",
        "orig", "phil", "pl", "pokrač", "pol", "port", "pozn", "př.kr", "př.n.l", "přel", "přeprac", "příl", "pseud",
        "pt", "red", "repr", "resp", "revid", "rkp", "roč", "roz", "rozš", "samost", "sect", "sest", "seš", "sign",
        "sl", "srv", "stol", "sv", "šk", "šk.ro", "špan", "tab", "t.č", "tis", "tj", "tř", "tzv", "univ", "uspoř",
        "vol", "vl.jm", "vs", "vyd", "vyobr", "zal", "zejm", "zkr", "zprac", "zvl", "n.p", "např", "než", "MUDr", "abl",
        "absol", "adj", "adv", "ak", "ak. sl", "akt", "alch", "amer", "anat", "angl", "anglosas", "arab", "arch",
        "archit", "arg", "astr", "astrol", "att", "bás", "belg", "bibl", "biol", "boh", "bot", "bulh", "círk", "csl",
        "č", "čas", "čes", "dat", "děj", "dep", "dět", "dial", "dór", "dopr", "dosl", "ekon", "epic", "etnonym",
        "eufem", "f", "fam", "fem", "fil", "film", "form", "fot", "fr", "fut", "fyz", "gen", "geogr", "geol", "geom",
        "germ", "gram", "hebr", "herald", "hist", "hl", "hovor", "hud", "hut", "chcsl", "chem", "ie", "imp", "impf",
        "ind", "indoevr", "inf", "instr", "interj", "ión", "iron", "it", "kanad", "katalán", "klas", "kniž", "komp",
        "konj", " ", "konkr", "kř", "kuch", "lat", "lék", "les", "lid", "lit", "liturg", "lok", "log", "m", "mat",
        "meteor", "metr", "mod", "ms", "mysl", "n", "náb", "námoř", "neklas", "něm", "nesklon", "nom", "ob", "obch",
        "obyč", "ojed", "opt", "part", "pas", "pejor", "pers", "pf", "pl", "plpf", " ", "práv", "prep", "předl",
        "přivl", "r", "rcsl", "refl", "reg", "rkp", "ř", "řec", "s", "samohl", "sg", "sl", "souhl", "spec", "srov",
        "stfr", "střv", "stsl", "subj", "subst", "superl", "sv", "sz", "táz", "tech", "telev", "teol", "trans",
        "typogr", "var", "vedl", "verb", "vl. jm", "voj", "vok", "vůb", "vulg", "výtv", "vztaž", "zahr", "zájm", "zast",
        "zejm", " ", "zeměd", "zkr", "zř", "mj", "dl", "atp", "sport", "Mgr", "horn", "MVDr", "JUDr", "RSDr", "Bc",
        "PhDr", "ThDr", "Ing", "aj", "apod", "PharmDr", "pomn", "ev", "slang", "nprap", "odp", "dop", "pol", "st",
        "stol", "p. n. l", "před n. l", "n. l", "př. Kr", "po Kr", "př. n. l", "odd", "RNDr", "tzv", "atd", "tzn",
        "resp", "tj", "p", "br", "č. j", "čj", "č. p", "čp", "a. s", "s. r. o", "spol. s r. o", "p. o", "s. p",
        "v. o. s", "k. s", "o. p. s", "o. s", "v. r", "v z", "ml", "vč", "kr", "mld", "hod", "popř", "ap", "event",
        "rus", "slov", "rum", "švýc", "P. T", "zvl", "hor", "dol", "S.O.S")),
    dutch -> NonBreakingPrefixes(
      prefixes = Set(
        // Any single upper case letter followed by a period is not at the end of a sentence (excluding "I"
        // occasionally, but we leave it in) usually upper case letters are initials in a name.
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
        "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        // List of titles. These are often followed by upper-case names, but do not indicate sentence breaks.
        "bacc", "bc", "bgen", "c.i", "dhr", "dr", "dr.h.c", "drs", "drs", "ds", "eint", "fa", "Fa", "fam", "gen",
        "genm", "ing", "ir", "jhr", "jkvr", "jr", "kand", "kol", "lgen", "lkol", "Lt", "maj", "Mej", "mevr", "Mme",
        "mr", "mr", "Mw", "o.b.s", "plv", "prof", "ritm", "tint", "Vz", "Z.D", "Z.D.H", "Z.E", "Z.Em", "Z.H", "Z.K.H",
        "Z.K.M", "Z.M", "z.v",
        // Miscellaneous symbols.
        "a.g.v", "bijv", "bijz", "bv", "d.w.z", "e.c", "e.g", "e.k", "ev", "i.p.v", "i.s.m", "i.t.t", "i.v.m", "m.a.w",
        "m.b.t", "m.b.v", "m.h.o", "m.i", "m.i.v", "v.w.t", "Nrs", "nrs"),
      numericPrefixes = Set("Nr", "nr")),
    english -> NonBreakingPrefixes(
      prefixes = Set(
        // Any single upper case letter followed by a period is not at the end of a sentence (excluding "I"
        // occasionally, but we leave it in) usually upper case letters are initials in a name.
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
        "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        // List of titles. These are often followed by upper-case names, but do not indicate sentence breaks.
        "Adj", "Adm", "Adv", "Asst", "Bart", "Bldg", "Brig", "Bros", "Capt", "Cmdr", "Col", "Comdr", "Con", "Corp",
        "Cpl", "DR", "Dr", "Drs", "Ens", "Gen", "Gov", "Hon", "Hr", "Hosp", "Insp", "Lt", "MM", "MR", "MRS", "MS",
        "Maj", "Messrs", "Mlle", "Mme", "Mr", "Mrs", "Ms", "Msgr", "Op", "Ord", "Pfc", "Ph", "Prof", "Pvt", "Rep",
        "Reps", "Res", "Rev", "Rt", "Sen", "Sens", "Sfc", "Sgt", "Sr", "St", "Supt", "Surg",
        // Miscellaneous symbols - we add period-ending items that never indicate breaks ("p.m." does not fall into this
        // category - it sometimes ends a sentence).
        "v", "vs", "i.e", "rev", "e.g", "Nos", "Nr",
        // Month abbreviations (note that "May" is also a full word and is thus not included here).
        "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"),
      numericPrefixes = Set(
        // Numbers only. These should only induce breaks when followed by a numeric sequence.
        "No", "Art", "pp")),
    finnish -> NonBreakingPrefixes(
      // This list is compiled from the omorfi (http://code.google.com/p/omorfi) database by Tommi A Pirinen.
      prefixes = Set(
        // Any single upper case letter followed by a period is not at the end of a sentence (excluding "I"
        // occasionally, but we leave it in) usually upper case letters are initials in a name.
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
        "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Å", "Ä", "Ö",
        // List of titles. These are often followed by upper-case names, but do not indicate sentence breaks.
        "alik", "alil", "amir", "apul", "apul.prof", "arkkit", "ass", "assist", "dipl", "dipl.arkkit", "dipl.ekon",
        "dipl.ins", "dipl.kielenk", "dipl.kirjeenv", "dipl.kosm", "dipl.urk", "dos", "erikoiseläinl", "erikoishammasl",
        "erikoisl", "erikoist", "ev.luutn", "evp", "fil", "ft", "hallinton", "hallintot", "hammaslääket", "jatk",
        "jääk", "kansaned", "kapt", "kapt.luutn", "kenr", "kenr.luutn", "kenr.maj", "kers", "kirjeenv", "kom",
        "kom.kapt", "komm", "konst", "korpr", "luutn", "maist", "maj", "Mr", "Mrs", "Ms", "M.Sc", "neuv", "nimim",
        "Ph.D", "prof", "puh.joht", "pääll", "res", "san", "siht", "suom", "sähköp", "säv", "toht", "toim", "toim.apul",
        "toim.joht", "toim.siht", "tuom", "ups", "vänr", "vääp", "ye.ups", "ylik", "ylil", "ylim", "ylimatr", "yliop",
        "yliopp", "ylip", "yliv",
        // Miscellaneous symbols - odd period-ending items that never indicate sentence breaks ("p.m." does not fall
        // into this category - it sometimes ends a sentence).
        "e.g", "ent", "esim", "huom", "i.e", "ilm", "l", "mm", "myöh", "nk", "nyk", "par", "po", "t", "v")),
    french -> NonBreakingPrefixes(
      prefixes = Set(
        // Any single upper case letter followed by a period is not at the end of a sentence (excluding "I"
        // occasionally, but we leave it in) usually upper case letters are initials in a name. No french words end in
        // single lower-case letters, and so we throw those in too.
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
        "W", "X", "Y", "Z", "#a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
        "s", "t", "u", "v", "w", "x", "y", "z",
        // Period-final abbreviation list for French.
        "A.C.N", "A.M", "art", "ann", "apr", "av", "auj", "lib", "B.P", "boul", "ca", "c.-à-d", "cf", "ch.-l", "chap",
        "contr", "C.P.I", "C.Q.F.D", "C.N", "C.N.S", "C.S", "dir", "éd", "e.g", "env", "al", "etc", "E.V", "ex", "fasc",
        "fém", "fig", "fr", "hab", "ibid", "id", "i.e", "inf", "LL.AA", "LL.AA.II", "LL.AA.RR", "LL.AA.SS", "L.D",
        "LL.EE", "LL.MM", "LL.MM.II.RR", "loc.cit", "masc", "MM", "ms", "N.B", "N.D.A", "N.D.L.R", "N.D.T", "n/réf",
        "NN.SS", "N.S", "N.D", "N.P.A.I", "p.c.c", "pl", "pp", "p.ex", "p.j", "P.S", "R.A.S", "R.-V", "R.P", "R.I.P",
        "SS", "S.S", "S.A", "S.A.I", "S.A.R", "S.A.S", "S.E", "sec", "sect", "sing", "S.M", "S.M.I.R", "sq", "sqq",
        "suiv", "sup", "suppl", "tél", "T.S.V.P", "vb", "vol", "vs", "X.O", "Z.I")),
    german -> NonBreakingPrefixes(
      prefixes = Set(
        // Any single upper case letter followed by a period is not at the end of a sentence (excluding "I"
        // occasionally, but we leave it in) usually upper case letters are initials in a name. No german words end in
        // single lower-case letters, and so we throw those in too.
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
        "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
        "s", "t", "u", "v", "w", "x", "y", "z",
        // Roman Numerals. A dot after one of these is not a sentence break in German.
        "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII",
        "XVIII", "XIX", "XX", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv",
        "xv", "xvi", "xvii", "xviii", "xix", "xx",
        // Titles and honorifics.
        "Adj", "Adm", "Adv", "Asst", "Bart", "Bldg", "Brig", "Bros", "Capt", "Cmdr", "Col", "Comdr", "Con", "Corp",
        "Cpl", "DR", "Dr", "Ens", "Gen", "Gov", "Hon", "Hosp", "Insp", "Lt", "MM", "MR", "MRS", "MS", "Maj", "Messrs",
        "Mlle", "Mme", "Mr", "Mrs", "Ms", "Msgr", "Op", "Ord", "Pfc", "Ph", "Prof", "Pvt", "Rep", "Reps", "Res", "Rev",
        "Rt", "Sen", "Sens", "Sfc", "Sgt", "Sr", "St", "Supt", "Surg",
        // Miscellaneous symbols.
        "Mio", "Mrd", "bzw", "v", "vs", "usw", "d.h", "z.B", "u.a", "etc", "Mrd", "MwSt", "ggf", "d.J", "D.h", "m.E",
        "vgl", "I.F", "z.T", "sogen", "ff", "u.E", "g.U", "g.g.A", "c.-à-d", "Buchst", "u.s.w", "sog", "u.ä", "Std",
        "evtl", "Zt", "Chr", "u.U", "o.ä", "Ltd", "b.A", "z.Zt", "spp", "sen", "SA", "k.o", "jun", "i.H.v", "dgl",
        "dergl", "Co", "zzt", "usf", "s.p.a", "Dkr", "Corp", "bzgl", "BSE", "Nos", "Nr", "ca", "Ca",
        // Ordinals are denoted with "." in German - "1." = "1st" in English.
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
        "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38",
        "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56",
        "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74",
        "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92",
        "93", "94", "95", "96", "97", "98", "99"),
      numericPrefixes = Set(
        // Numbers only. These should only induce breaks when followed by a numeric sequence.
        "No", "Art", "pp")),
    greek -> NonBreakingPrefixes(
      prefixes = Set(
        // Single letters in upper-case are usually abbreviations of names.
        "Α", "Β", "Γ", "Δ", "Ε", "Ζ", "Η", "Θ", "Ι", "Κ", "Λ", "Μ", "Ν", "Ξ",
        "Ο", "Π", "Ρ", "Σ", "Τ", "Υ", "Φ", "Χ", "Ψ", "Ω",
        // Includes abbreviations for the Greek language compiled from various sources (Greek grammar books, Greek
        // language related web content).
        "Άθαν", "Έγχρ", "Έκθ", "Έσδ", "Έφ", "Όμ", "Α΄Έσδρ", "Α΄Έσδ", "Α΄Βασ", "Α΄Θεσ", "Α΄Ιω", "Α΄Κορινθ", "Α΄Κορ",
        "Α΄Μακκ", "Α΄Μακ", "Α΄Πέτρ", "Α΄Πέτ", "Α΄Παραλ", "Α΄Πε", "Α΄Σαμ", "Α΄Τιμ", "Α΄Χρον", "Α΄Χρ", "Α.Β.Α", "Α.Β",
        "Α.Ε", "Α.Κ.Τ.Ο", "Αέθλ", "Αέτ", "Αίλ.Δ", "Αίλ.Τακτ", "Αίσ", "Αββακ", "Αβυδ", "Αβ", "Αγάκλ", "Αγάπ",
        "Αγάπ.Αμαρτ.Σ", "Αγάπ.Γεωπ", "Αγαθάγγ", "Αγαθήμ", "Αγαθιν", "Αγαθοκλ", "Αγαθρχ", "Αγαθ", "Αγαθ.Ιστ", "Αγαλλ",
        "Αγαπητ", "Αγγ", "Αγησ", "Αγλ", "Αγορ.Κ", "Αγρο.Κωδ", "Αγρ.Εξ", "Αγρ.Κ", "Αγ.Γρ", "Αδριαν", "Αδρ", "Αετ",
        "Αθάν", "Αθήν", "Αθήν.Επιγρ", "Αθήν.Επιτ", "Αθήν.Ιατρ", "Αθήν.Μηχ", "Αθανάσ", "Αθαν", "Αθηνί", "Αθηναγ",
        "Αθηνόδ", "Αθ", "Αθ.Αρχ", "Αιλ", "Αιλ.Επιστ", "Αιλ.ΖΙ", "Αιλ.ΠΙ", "Αιλ.απ", "Αιμιλ", "Αιν.Γαζ", "Αιν.Τακτ",
        "Αισχίν", "Αισχίν.Επιστ", "Αισχ", "Αισχ.Αγαμ", "Αισχ.Αγ", "Αισχ.Αλ", "Αισχ.Ελεγ", "Αισχ.Επτ.Θ", "Αισχ.Ευμ",
        "Αισχ.Ικέτ", "Αισχ.Ικ", "Αισχ.Περσ", "Αισχ.Προμ.Δεσμ", "Αισχ.Πρ", "Αισχ.Χοηφ", "Αισχ.Χο", "Αισχ.απ", "ΑιτΕ",
        "Αιτ", "Αλκ", "Αλχιας", "Αμ.Π.Ο", "Αμβ", "Αμμών", "Αμ.", "Αν.Πειθ.Συμβ.Δικ", "Ανακρ", "Ανακ", "Αναμν.Τόμ",
        "Αναπλ", "Ανδ", "Ανθλγος", "Ανθστης", "Αντισθ", "Ανχης", "Αν", "Αποκ", "Απρ", "Απόδ", "Απόφ", "Απόφ.Νομ",
        "Απ", "Απ.Δαπ", "Απ.Διατ", "Απ.Επιστ", "Αριθ", "Αριστοτ", "Αριστοφ", "Αριστοφ.Όρν", "Αριστοφ.Αχ",
        "Αριστοφ.Βάτρ", "Αριστοφ.Ειρ", "Αριστοφ.Εκκλ", "Αριστοφ.Θεσμ", "Αριστοφ.Ιππ", "Αριστοφ.Λυσ", "Αριστοφ.Νεφ",
        "Αριστοφ.Πλ", "Αριστοφ.Σφ", "Αριστ", "Αριστ.Αθ.Πολ", "Αριστ.Αισθ", "Αριστ.Αν.Πρ", "Αριστ.Ζ.Ι", "Αριστ.Ηθ.Ευδ",
        "Αριστ.Ηθ.Νικ", "Αριστ.Κατ", "Αριστ.Μετ", "Αριστ.Πολ", "Αριστ.Φυσιογν", "Αριστ.Φυσ", "Αριστ.Ψυχ", "Αριστ.Ρητ",
        "Αρμεν", "Αρμ", "Αρχ.Εκ.Καν.Δ", "Αρχ.Ευβ.Μελ", "Αρχ.Ιδ.Δ", "Αρχ.Νομ", "Αρχ.Ν", "Αρχ.Π.Ε", "Αρ", "Αρ.Φορ.Μητρ",
        "Ασμ", "Ασμ.ασμ", "Αστ.Δ", "Αστ.Χρον", "Ασ", "Ατομ.Γνωμ", "Αυγ", "Αφρ", "Αχ.Νομ", "Α", "Α.Εγχ.Π",
        "Α.Κ.΄Υδρας", "Β΄Έσδρ", "Β΄Έσδ", "Β΄Βασ", "Β΄Θεσ", "Β΄Ιω", "Β΄Κορινθ", "Β΄Κορ", "Β΄Μακκ", "Β΄Μακ",
        "Β΄Πέτρ", "Β΄Πέτ", "Β΄Πέ", "Β΄Παραλ", "Β΄Σαμ", "Β΄Τιμ", "Β΄Χρον", "Β΄Χρ", "Β.Ι.Π.Ε", "Β.Κ.Τ", "Β.Κ.Ψ.Β",
        "Β.Μ", "Β.Ο.Α.Κ", "Β.Ο.Α", "Β.Ο.Δ", "Βίβλ", "Βαρ", "ΒεΘ", "Βι.Περ", "Βιπερ", "Βιργ", "Βλγ", "Βούλ",
        "Βρ", "Γ΄Βασ", "Γ΄Μακκ", "ΓΕΝμλ", "Γέν", "Γαλ", "Γεν", "Γλ", "Γν.Ν.Σ.Κρ", "Γνωμ", "Γν", "Γράμμ", "Γρηγ.Ναζ",
        "Γρηγ.Νύσ", "Γ Νοσ", "Γ' Ογκολ", "Γ.Ν", "Δ΄Βασ", "Δ.Β", "Δ.Δίκη", "Δ.Δίκ", "Δ.Ε.Σ", "Δ.Ε.Φ.Α",
        "Δ.Ε.Φ", "Δ.Εργ.Ν", "Δαμ", "Δαμ.μνημ.έργ", "Δαν", "Δασ.Κ", "Δεκ", "Δελτ.Δικ.Ε.Τ.Ε", "Δελτ.Νομ",
        "Δελτ.Συνδ.Α.Ε", "Δερμ", "Δευτ", "Δεύτ", "Δημοσθ", "Δημόκρ", "Δι.Δικ", "Διάτ", "Διαιτ.Απ", "Διαιτ",
        "Διαρκ.Στρατ", "Δικ", "Διοίκ.Πρωτ", "ΔιοικΔνη", "Διοικ.Εφ", "Διον.Αρ", "Διόρθ.Λαθ", "Δ.κ.Π", "Δνη", "Δν",
        "Δογμ.Όρος", "Δρ", "Δ.τ.Α", "Δτ", "ΔωδΝομ", "Δ.Περ", "Δ.Στρ", "ΕΔΠολ", "ΕΕυρΚ", "ΕΙΣ", "ΕΝαυτΔ", "ΕΣΑμΕΑ",
        "ΕΣΘ", "ΕΣυγκΔ", "ΕΤρΑξΧρΔ", "Ε.Φ.Ε.Τ", "Ε.Φ.Ι", "Ε.Φ.Ο.Επ.Α", "Εβδ", "Εβρ", "Εγκύκλ.Επιστ", "Εγκ", "Εε.Αιγ",
        "Εθν.Κ.Τ", "Εθν", "Ειδ.Δικ.Αγ.Κακ", "Εικ", "Ειρ.Αθ", "Ειρην.Αθ", "Ειρην", "Έλεγχ", "Ειρ", "Εισ.Α.Π", "Εισ.Ε",
        "Εισ.Ν.Α.Κ", "Εισ.Ν.Κ.Πολ.Δ", "Εισ.Πρωτ", "Εισηγ.Έκθ", "Εισ", "Εκκλ", "Εκκ", "Εκ",
        "Ελλ.Δνη", "Εν.Ε", "Εξ", "Επ.Αν", "Επ.Εργ.Δ", "Επ.Εφ", "Επ.Κυπ.Δ", "Επ.Μεσ.Αρχ", "Επ.Νομ", "Επίκτ", "Επίκ",
        "Επι.Δ.Ε", "Επιθ.Ναυτ.Δικ", "Επικ", "Επισκ.Ε.Δ", "Επισκ.Εμπ.Δικ", "Επιστ.Επετ.Αρμ", "Επιστ.Επετ", "Επιστ.Ιερ",
        "Επιτρ.Προστ.Συνδ.Στελ", "Επιφάν", "Επτ.Εφ", "Επ.Ιρ", "Επ.Ι", "Εργ.Ασφ.Νομ", "Ερμ.Α.Κ", "Ερμη.Σ", "Εσθ",
        "Εσπερ", "Ετρ.Δ", "Ευκλ", "Ευρ.Δ.Δ.Α", "Ευρ.Σ.Δ.Α", "Ευρ.ΣτΕ", "Ευρατόμ", "Ευρ.Άλκ", "Ευρ.Ανδρομ", "Ευρ.Βάκχ",
        "Ευρ.Εκ", "Ευρ.Ελ", "Ευρ.Ηλ", "Ευρ.Ηρακ", "Ευρ.Ηρ", "Ευρ.Ηρ.Μαιν", "Ευρ.Ικέτ",
        "Ευρ.Ιππόλ", "Ευρ.Ιφ.Α", "Ευρ.Ιφ.Τ", "Ευρ.Ι.Τ", "Ευρ.Κύκλ", "Ευρ.Μήδ", "Ευρ.Ορ", "Ευρ.Ρήσ", "Ευρ.Τρωάδ",
        "Ευρ.Φοίν", "Εφ.Αθ", "Εφ.Εν", "Εφ.Επ", "Εφ.Θρ", "Εφ.Θ", "Εφ.Ι", "Εφ.Κερ", "Εφ.Κρ", "Εφ.Λ", "Εφ.Ν", "Εφ.Πατ",
        "Εφ.Πειρ", "Εφαρμ.Δ.Δ", "Εφαρμ", "Εφεσ", "Εφημ", "Εφ", "Ζαχ", "Ζιγ", "Ζυ", "Ζχ", "ΗΕ.Δ",
        "Ημερ", "Ηράκλ", "Ηροδ", "Ησίοδ", "Ησ", "Η.Ε.Γ", "ΘΗΣ", "ΘΡ", "Θαλ", "Θεοδ", "Θεοφ", "Θεσ", "Θεόδ.Μοψ",
        "Θεόκρ", "Θεόφιλ", "Θουκ", "Θρ", "Θρ.Ε", "Θρ.Ιερ", "Θρ.Ιρ", "Ιακ", "Ιαν", "Ιβ", "Ιδθ", "Ιδ", "Ιεζ", "Ιερ",
        "Ιζ", "Ιησ", "Ιησ.Ν", "Ικ", "Ιλ", "Ιν", "Ιουδ", "Ιουστ", "Ιούδα", "Ιούλ", "Ιούν", "Ιπποκρ", "Ιππόλ", "Ιρ",
        "Ισίδ.Πηλ", "Ισοκρ", "Ισ.Ν", "Ιωβ", "Ιωλ", "Ιων", "Ιω", "ΚΟΣ", "ΚΟ.ΜΕ.ΚΟΝ", "ΚΠοινΔ", "ΚΠολΔ", "ΚαΒ",
        "Καλ", "Καλ.Τέχν", "ΚανΒ", "Καν.Διαδ", "Κατάργ", "Κλ", "ΚοινΔ", "Κολσ", "Κολ", "Κον", "Κορ", "Κος",
        "ΚριτΕπιθ", "ΚριτΕ", "Κριτ", "Κρ", "ΚτΒ", "ΚτΕ", "ΚτΠ", "Κυβ", "Κυπρ", "Κύριλ.Αλεξ", "Κύριλ.Ιερ", "Λεβ",
        "Λεξ.Σουίδα", "Λευϊτ", "Λευ", "Λκ", "Λογ", "ΛουκΑμ", "Λουκιαν", "Λουκ.Έρωτ", "Λουκ.Ενάλ.Διάλ", "Λουκ.Ερμ",
        "Λουκ.Εταιρ.Διάλ", "Λουκ.Ε.Δ", "Λουκ.Θε.Δ", "Λουκ.Ικ.", "Λουκ.Ιππ", "Λουκ.Λεξιφ", "Λουκ.Μεν", "Λουκ.Μισθ.Συν",
        "Λουκ.Ορχ", "Λουκ.Περ", "Λουκ.Συρ", "Λουκ.Τοξ", "Λουκ.Τυρ", "Λουκ.Φιλοψ", "Λουκ.Φιλ", "Λουκ.Χάρ", "Λουκ.",
        "Λουκ.Αλ", "Λοχ", "Λυδ", "Λυκ", "Λυσ", "Λωζ", "Λ1", "Λ2", "ΜΟΕφ", "Μάρκ", "Μέν",
        "Μαλ", "Ματθ", "Μα", "Μιχ", "Μκ", "Μλ", "Μμ", "Μον.Δ.Π", "Μον.Πρωτ", "Μον", "Μρ", "Μτ", "Μχ", "Μ.Βασ", "Μ.Πλ",
        "ΝΑ", "Ναυτ.Χρον", "Να", "Νδικ", "Νεεμ", "Νε", "Νικ", "ΝκΦ", "Νμ", "ΝοΒ", "Νομ.Δελτ.Τρ.Ελ", "Νομ.Δελτ",
        "Νομ.Σ.Κ", "Νομ.Χρ", "Νομ", "Νομ.Διεύθ", "Νοσ", "Ντ", "Νόσων", "Ν1", "Ν2", "Ν3", "Ν4", "Νtot",
        "Ξενοφ", "Ξεν", "Ξεν.Ανάβ", "Ξεν.Απολ", "Ξεν.Απομν", "Ξεν.Απομ", "Ξεν.Ελλ", "Ξεν.Ιέρ", "Ξεν.Ιππαρχ",
        "Ξεν.Ιππ", "Ξεν.Κυρ.Αν", "Ξεν.Κύρ.Παιδ", "Ξεν.Κ.Π", "Ξεν.Λακ.Πολ", "Ξεν.Οικ", "Ξεν.Προσ", "Ξεν.Συμπόσ",
        "Ξεν.Συμπ", "Ο΄", "Οβδ", "Οβ", "ΟικΕ", "Οικ", "Οικ.Πατρ", "Οικ.Σύν.Βατ", "Ολομ", "Ολ", "Ολ.Α.Π", "Ομ.Ιλ",
        "Ομ.Οδ", "ΟπΤοιχ", "Οράτ", "Ορθ", "ΠΡΟ.ΠΟ", "Πίνδ", "Πίνδ.Ι", "Πίνδ.Νεμ", "Πίνδ.Ν", "Πίνδ.Ολ", "Πίνδ.Παθ",
        "Πίνδ.Πυθ", "Πίνδ.Π", "ΠαγΝμλγ", "Παν", "Παρμ", "Παροιμ", "Παρ", "Παυσ", "Πειθ.Συμβ", "ΠειρΝ", "Πελ",
        "ΠεντΣτρ", "Πεντ", "Πεντ.Εφ", "ΠερΔικ", "Περ.Γεν.Νοσ", "Πετ", "Πλάτ", "Πλάτ.Αλκ", "Πλάτ.Αντ", "Πλάτ.Αξίοχ",
        "Πλάτ.Απόλ", "Πλάτ.Γοργ", "Πλάτ.Ευθ", "Πλάτ.Θεαίτ", "Πλάτ.Κρατ", "Πλάτ.Κριτ", "Πλάτ.Λύσ", "Πλάτ.Μεν",
        "Πλάτ.Νόμ", "Πλάτ.Πολιτ", "Πλάτ.Πολ", "Πλάτ.Πρωτ", "Πλάτ.Σοφ.", "Πλάτ.Συμπ", "Πλάτ.Τίμ", "Πλάτ.Φαίδρ",
        "Πλάτ.Φιλ", "Πλημ", "Πλούτ", "Πλούτ.Άρατ", "Πλούτ.Αιμ", "Πλούτ.Αλέξ", "Πλούτ.Αλκ", "Πλούτ.Αντ", "Πλούτ.Αρτ",
        "Πλούτ.Ηθ", "Πλούτ.Θεμ", "Πλούτ.Κάμ", "Πλούτ.Καίσ", "Πλούτ.Κικ", "Πλούτ.Κράσ", "Πλούτ.Κ",
        "Πλούτ.Λυκ", "Πλούτ.Μάρκ", "Πλούτ.Μάρ", "Πλούτ.Περ", "Πλούτ.Ρωμ", "Πλούτ.Σύλλ", "Πλούτ.Φλαμ", "Πλ",
        "Ποιν.Δικ", "Ποιν.Δ", "Ποιν.Ν", "Ποιν.Χρον", "Ποιν.Χρ", "Πολ.Δ", "Πολ.Πρωτ", "Πολ", "Πολ.Μηχ", "Πολ.Μ",
        "Πρακτ.Αναθ", "Πρακτ.Ολ", "Πραξ", "Πρμ", "Πρξ", "Πρωτ", "Πρ", "Πρ.Αν", "Πρ.Λογ", "Πταισμ", "Πυρ.Καλ",
        "Πόλη", "Π.Δ", "Π.Δ.Άσμ", "ΡΜ.Ε", "Ρθ", "Ρμ", "Ρωμ", "ΣΠλημ", "Σαπφ", "Σειρ", "Σολ", "Σοφ", "Σοφ.Αντιγ",
        "Σοφ.Αντ", "Σοφ.Αποσ", "Σοφ.Απ", "Σοφ.Ηλέκ", "Σοφ.Ηλ", "Σοφ.Οιδ.Κολ", "Σοφ.Οιδ.Τύρ", "Σοφ.Ο.Τ", "Σοφ.Σειρ",
        "Σοφ.Σολ", "Σοφ.Τραχ", "Σοφ.Φιλοκτ", "Σρ", "Σ.τ.Ε", "Σ.τ.Π", "Στρ.Π.Κ", "Στ.Ευρ", "Συζήτ", "Συλλ.Νομολ",
        "Συλ.Νομ", "ΣυμβΕπιθ", "Συμπ.Ν", "Συνθ.Αμ", "Συνθ.Ε.Ε", "Συνθ.Ε.Κ", "Συνθ.Ν", "Σφν", "Σφ", "Σφ.Σλ",
        "Σχ.Πολ.Δ", "Σχ.Συντ.Ε", "Σωσ", "Σύντ", "Σ.Πληρ", "ΤΘ", "ΤΣ.Δ", "Τίτ", "Τβ", "Τελ.Ενημ", "Τελ.Κ", "Τερτυλ",
        "Τιμ", "Τοπ.Α", "Τρ.Ο", "Τριμ", "Τριμ.Πλ", "Τρ.Πλημ", "Τρ.Π.Δ", "Τ.τ.Ε", "Ττ", "Τωβ", "Υγ", "Υπερ", "Υπ",
        "Υ.Γ", "Φιλήμ", "Φιλιπ", "Φιλ", "Φλμ", "Φλ", "Φορ.Β", "Φορ.Δ.Ε", "Φορ.Δνη", "Φορ.Δ", "Φορ.Επ", "Φώτ",
        "Χρ.Ι.Δ", "Χρ.Ιδ.Δ", "Χρ.Ο", "Χρυσ", "Ψήφ", "Ψαλμ", "Ψαλ", "Ψλ", "Ωριγ", "Ωσ", "Ω.Ρ.Λ", "άγν", "άγν.ετυμολ",
        "άγ", "άκλ", "άνθρ", "άπ", "άρθρ", "άρν", "άρ", "άτ", "άψ", "ά", "έκδ", "έκφρ", "έμψ", "ένθ.αν", "έτ", "έ.α",
        "ίδ", "αβεστ", "αβησσ", "αγγλ", "αγγ", "αδημ", "αεροναυτ", "αερον", "αεροπ",
        "αθλητ", "αθλ", "αθροιστ", "αιγυπτ", "αιγ", "αιτιολ", "αιτ", "αι", "ακαδ", "ακκαδ", "αλβ", "αλλ",
        "αλφαβητ", "αμα", "αμερικ", "αμερ", "αμετάβ", "αμτβ", "αμφιβ", "αμφισβ", "αμφ", "αμ", "ανάλ", "ανάπτ",
        "ανάτ", "αναβ", "αναδαν", "αναδιπλασ", "αναδιπλ", "αναδρ", "αναλ", "αναν", "ανασυλλ", "ανατολ", "ανατομ",
        "ανατυπ", "ανατ", "αναφορ", "αναφ", "ανα.ε", "ανδρων", "ανθρωπολ", "ανθρωπ", "ανθ", "ανομ", "αντίτ",
        "αντδ", "αντιγρ", "αντιθ", "αντικ", "αντιμετάθ", "αντων", "αντ", "ανωτ", "ανόργ", "ανών", "αορ", "απαρέμφ",
        "απαρφ", "απαρχ", "απαρ", "απλολ", "απλοπ", "αποβ", "αποηχηροπ", "αποθ", "αποκρυφ", "αποφ", "απρμφ",
        "απρφ", "απρόσ", "απόδ", "απόλ", "απόσπ", "απόφ", "αραβοτουρκ", "αραβ", "αραμ", "αρβαν", "αργκ", "αριθμτ",
        "αριθμ", "αριθ", "αρκτικόλ", "αρκ", "αρμεν", "αρμ", "αρνητ", "αρσ", "αρχαιολ", "αρχιτεκτ", "αρχιτ", "αρχκ",
        "αρχ", "αρωμουν", "αρωμ", "αρ", "αρ.μετρ", "αρ.φ", "ασσυρ", "αστρολ", "αστροναυτ", "αστρον", "αττ",
        "αυστραλ", "αυτοπ", "αυτ", "αφγαν", "αφηρ", "αφομ", "αφρικ", "αχώρ", "αόρ", "α.α", "α/α", "α0", "βαθμ",
        "βαθ", "βαπτ", "βασκ", "βεβαιωτ", "βεβ", "βεδ", "βενετ", "βεν", "βερβερ", "βιβλγρ", "βιολ", "βιομ",
        "βιοχημ", "βιοχ", "βλάχ", "βλ", "βλ.λ", "βοταν", "βοτ", "βουλγαρ", "βουλγ", "βούλ", "βραζιλ", "βρετον",
        "βόρ", "γαλλ", "γενικότ", "γενοβ", "γεν", "γερμαν", "γερμ", "γεωγρ", "γεωλ", "γεωμετρ", "γεωμ", "γεωπ",
        "γεωργ", "γλυπτ", "γλωσσολ", "γλωσσ", "γλ", "γνμδ", "γνμ", "γνωμ", "γοτθ", "γραμμ", "γραμ", "γρμ", "γρ",
        "γυμν", "δίδες", "δίκ", "δίφθ", "δαν", "δεικτ", "δεκατ", "δηλ", "δημογρ", "δημοτ", "δημώδ", "δημ", "διάγρ",
        "διάκρ", "διάλεξ", "διάλ", "διάσπ", "διαλεκτ", "διατρ", "διαφ", "διαχ", "διδα", "διεθν", "διεθ", "δικον",
        "διστ", "δισύλλ", "δισ", "διφθογγοπ", "δογμ", "δολ", "δοτ", "δρμ", "δρχ", "δρ(α)", "δωρ", "δ", "εβρ",
        "εγκλπ", "εδ", "εθνολ", "εθν", "ειδικότ", "ειδ", "ειδ.β", "εικ", "ειρ", "εισ", "εκατοστμ", "εκατοστ",
        "εκατστ.2", "εκατστ.3", "εκατ", "εκδ", "εκκλησ", "εκκλ", "εκ", "ελλην", "ελλ", "ελνστ", "ελπ", "εμβ",
        "εμφ", "εναλλ", "ενδ", "ενεργ", "ενεστ", "ενικ", "ενν", "εν", "εξέλ", "εξακολ", "εξομάλ", "εξ", "εο",
        "επέκτ", "επίδρ", "επίθ", "επίρρ", "επίσ", "επαγγελμ", "επανάλ", "επανέκδ", "επιθ", "επικ", "επιμ",
        "επιρρ", "επιστ", "επιτατ", "επιφ", "επών", "επ", "εργ", "ερμ", "ερρινοπ", "ερωτ", "ετρουσκ", "ετυμ", "ετ",
        "ευφ", "ευχετ", "εφ", "εύχρ", "ε.α", "ε/υ", "ε0", "ζωγρ", "ζωολ", "ηθικ", "ηθ", "ηλεκτρολ", "ηλεκτρον",
        "ηλεκτρ", "ημίτ", "ημίφ", "ημιφ", "ηχηροπ", "ηχηρ", "ηχομιμ", "ηχ", "η", "θέατρ", "θεολ", "θετ", "θηλ",
        "θρακ", "θρησκειολ", "θρησκ", "θ", "ιαπων", "ιατρ", "ιδιωμ", "ιδ", "ινδ", "ιραν", "ισπαν", "ιστορ", "ιστ",
        "ισχυροπ", "ιταλ", "ιχθυολ", "ιων", "κάτ", "καθ", "κακοσ", "καν", "καρ", "κατάλ", "κατατ", "κατωτ", "κατ",
        "κα", "κελτ", "κεφ", "κινεζ", "κινημ", "κλητ", "κλιτ", "κλπ", "κλ", "κν", "κοινωνιολ", "κοινων", "κοπτ",
        "κουτσοβλαχ", "κουτσοβλ", "κπ", "κρ.γν", "κτγ", "κτην", "κτητ", "κτλ", "κτ", "κυριολ", "κυρ", "κύρ", "κ",
        "κ.ά", "κ.ά.π", "κ.α", "κ.εξ", "κ.επ", "κ.ε", "κ.λπ", "κ.λ.π", "κ.ού.κ", "κ.ο.κ", "κ.τ.λ", "κ.τ.τ", "κ.τ.ό",
        "λέξ", "λαογρ", "λαπ", "λατιν", "λατ", "λαϊκότρ", "λαϊκ", "λετ", "λιθ", "λογιστ", "λογοτ", "λογ",
        "λουβ", "λυδ", "λόγ", "λ", "λ.χ", "μέλλ", "μέσ", "μαθημ", "μαθ", "μαιευτ", "μαλαισ", "μαλτ", "μαμμων",
        "μεγεθ", "μεε", "μειωτ", "μελ", "μεξ", "μεσν", "μεσογ", "μεσοπαθ", "μεσοφ", "μετάθ", "μεταβτ", "μεταβ",
        "μετακ", "μεταπλ", "μεταπτωτ", "μεταρ", "μεταφορ", "μετβ", "μετεπιθ", "μετεπιρρ", "μετεωρολ", "μετεωρ",
        "μετον", "μετουσ", "μετοχ", "μετρ", "μετ", "μητρων", "μηχανολ", "μηχ", "μικροβιολ", "μογγολ", "μορφολ",
        "μουσ", "μπενελούξ", "μσνλατ", "μσν", "μτβ", "μτγν", "μτγ", "μτφρδ", "μτφρ", "μτφ", "μτχ", "μυθ", "μυκην",
        "μυκ", "μφ", "μ", "μ.ε", "μ.μ", "μ.π.ε", "μ.π.π", "μ0", "ναυτ", "νεοελλ", "νεολατιν", "νεολατ", "νεολ",
        "νεότ", "νλατ", "νομ", "νορβ", "νοσ", "νότ", "ν", "ξ.λ", "οικοδ", "οικολ", "οικον", "οικ", "ολλανδ", "ολλ",
        "ομηρ", "ομόρρ", "ονομ", "ον", "οπτ", "ορθογρ", "ορθ", "οριστ", "ορυκτολ", "ορυκτ", "ορ", "οσετ", "οσκ",
        "ουαλ", "ουγγρ", "ουδ", "ουσιαστικοπ", "ουσιαστ", "ουσ", "πίν", "παθητ", "παθολ", "παθ", "παιδ",
        "παλαιοντ", "παλαιότ", "παλ", "παππων", "παράγρ", "παράγ", "παράλλ", "παράλ", "παραγ", "παρακ", "παραλ",
        "παραπ", "παρατ", "παρβ", "παρετυμ", "παροξ", "παρων", "παρωχ", "παρ", "παρ.φρ", "πατριδων", "πατρων",
        "πβ", "περιθ", "περιλ", "περιφρ", "περσ", "περ", "πιθ", "πληθ", "πληροφ", "ποδ", "ποιητ", "πολιτ",
        "πολλαπλ", "πολ", "πορτογαλ", "πορτ", "ποσ", "πρακριτ", "πρβλ", "πρβ", "πργ", "πρκμ", "πρκ", "πρλ",
        "προέλ", "προβηγκ", "προελλ", "προηγ", "προθεμ", "προπαραλ", "προπαροξ", "προπερισπ", "προσαρμ",
        "προσηγορ", "προσταχτ", "προστ", "προσφών", "προσ", "προτακτ", "προτ.Εισ", "προφ", "προχωρ", "πρτ", "πρόθ",
        "πρόσθ", "πρόσ", "πρότ", "πρ", "πρ.Εφ", "πτ", "πυ", "π", "π.Χ", "π.μ", "π.χ", "ρήμ", "ρίζ", "ρηματ",
        "ρητορ", "ριν", "ρουμ", "ρωμ", "ρωσ", "ρ", "σανσκρ", "σαξ", "σελ", "σερβοκρ", "σερβ", "σημασιολ", "σημδ",
        "σημειολ", "σημερ", "σημιτ", "σημ", "σκανδ", "σκυθ", "σκωπτ", "σλαβ", "σλοβ", "σουηδ", "σουμερ", "σουπ",
        "σπάν", "σπανιότ", "σπ", "σσ", "στατ", "στερ", "στιγμ", "στιχ", "στρέμ", "στρατιωτ", "στρατ", "στ", "συγγ",
        "συγκρ", "συγκ", "συμπερ", "συμπλεκτ", "συμπλ", "συμπροφ", "συμφυρ", "συμφ", "συνήθ", "συνίζ", "συναίρ",
        "συναισθ", "συνδετ", "συνδ", "συνεκδ", "συνηρ", "συνθετ", "συνθ", "συνοπτ", "συντελ", "συντομογρ", "συντ",
        "συν", "συρ", "σχημ", "σχ", "σύγκρ", "σύμπλ", "σύμφ", "σύνδ", "σύνθ", "σύντμ", "σύντ", "σ", "σ.π", "σ/β",
        "τακτ", "τελ", "τετρ", "τετρ.μ", "τεχνλ", "τεχνολ", "τεχν", "τεύχ", "τηλεπικ", "τηλεόρ", "τιμ", "τιμ.τομ",
        "τοΣ", "τον", "τοπογρ", "τοπων", "τοπ", "τοσκ", "τουρκ", "τοχ", "τριτοπρόσ", "τροποπ", "τροπ", "τσεχ",
        "τσιγγ", "ττ", "τυπ", "τόμ", "τόνν", "τ", "τ.μ", "τ.χλμ", "υβρ", "υπερθ", "υπερσ", "υπερ", "υπεύθ", "υποθ",
        "υποκορ", "υποκ", "υποσημ", "υποτ", "υποφ", "υποχωρ", "υπόλ", "υπόχρ", "υπ", "υστλατ", "υψόμ", "υψ", "φάκ",
        "φαρμακολ", "φαρμ", "φιλολ", "φιλοσ", "φιλοτ", "φινλ", "φοινικ", "φράγκ", "φρανκον", "φριζ", "φρ", "φυλλ",
        "φυσιολ", "φυσ", "φωνηεντ", "φωνητ", "φωνολ", "φων", "φωτογρ", "φ", "φ.τ.μ", "χαμιτ", "χαρτόσ", "χαρτ",
        "χασμ", "χαϊδ", "χγφ", "χειλ", "χεττ", "χημ", "χιλ", "χλγρ", "χλγ", "χλμ", "χλμ.2", "χλμ.3", "χλσγρ",
        "χλστγρ", "χλστμ", "χλστμ.2", "χλστμ.3", "χλ", "χργρ", "χρημ", "χρον", "χρ", "χφ", "χ.ε", "χ.κ", "χ.ο", "χ.σ",
        "χ.τ", "χ.χ", "ψευδ", "ψυχαν", "ψυχιατρ", "ψυχολ", "ψυχ", "ωκεαν", "όμ", "όν", "όπ.παρ", "όπ.π",
        "ό.π", "ύψ", "1Βσ", "1Εσ", "1Θσ", "1Ιν", "1Κρ", "1Μκ", "1Πρ", "1Πτ", "1Τμ", "2Βσ", "2Εσ", "2Θσ", "2Ιν",
        "2Κρ", "2Μκ", "2Πρ", "2Πτ", "2Τμ", "3Βσ", "3Ιν", "3Μκ", "4Βσ")),
    hungarian -> NonBreakingPrefixes(
      prefixes = Set(
        // Any single upper case letter followed by a period is not at the end of a sentence (excluding "I"
        // occasionally, but we leave it in) usually upper case letters are initials in a name.
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
        "S", "T", "U", "V", "W", "X", "Y", "Z", "Á", "É", "Í", "Ó", "Ö", "Ő", "Ú", "Ü", "Ű",
        // List of titles. These are often followed by upper-case names, but do not indicate sentence breaks.
        "Dr", "dr", "kb", "Kb", "vö", "Vö", "pl", "Pl", "ca", "Ca", "min", "Min", "max", "Max", "ún",
        "Ún", "prof", "Prof", "de", "De", "du", "Du", "Szt", "St"),
      numericPrefixes = Set(
        // Month name abbreviations.
        "jan", "Jan", "Feb", "feb", "márc", "Márc", "ápr", "Ápr", "máj", "Máj", "jún", "Jún", "Júl", "júl", "aug",
        "Aug", "Szept", "szept", "okt", "Okt", "nov", "Nov", "dec", "Dec",
        // Other abbreviations.
        "tel", "Tel", "Fax", "fax")),
    icelandic -> NonBreakingPrefixes(
      prefixes = Set(
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
        "w", "x", "y", "z", "^", "í", "á", "ó", "æ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "ab.fn", "a.fn", "afs", "al", "alm", "alg",
        "andh", "ath", "aths", "atr", "ao", "au", "aukaf", "áfn", "áhrl.s", "áhrs", "ákv.gr", "ákv", "bh", "bls", "dr",
        "e.Kr", "et", "ef", "efn", "ennfr", "eink", "end", "e.st", "erl", "fél", "fskj", "fh", "f.hl", "físl", "fl",
        "fn", "fo", "forl", "frb", "frl", "frh", "frt", "fsl", "fsh", "fs", "fsk", "fst", "f.Kr", "ft", "fv", "fyrrn",
        "fyrrv", "germ", "gm", "gr", "hdl", "hdr", "hf", "hl", "hlsk", "hljsk", "hljv", "hljóðv", "hr", "hv", "hvk",
        "holl", "Hos", "höf", "hk", "hrl", "ísl", "kaf", "kap", "Khöfn", "kk", "kg", "kk", "km", "kl", "klst", "kr",
        "kt", "kgúrsk", "kvk", "leturbr", "lh", "lh.nt", "lh.þt", "lo", "ltr", "mlja", "mljó", "millj", "mm", "mms",
        "m.fl", "miðm", "mgr", "mst", "mín", "nf", "nh", "nhm", "nl", "nk", "nmgr", "no", "núv", "nt", "o.áfr",
        "o.m.fl", "ohf", "o.fl", "o.s.frv", "ófn", "ób", "óákv.gr", "óákv", "pfn", "PR", "pr", "Ritstj", "Rvík", "Rvk",
        "samb", "samhlj", "samn", "samn", "sbr", "sek", "sérn", "sf", "sfn", "sh", "sfn", "sh", "s.hl", "sk", "skv",
        "sl", "sn", "so", "ss.us", "s.st", "samþ", "sbr", "shlj", "sign", "skál", "st", "st.s", "stk", "sþ", "teg",
        "tbl", "tfn", "tl", "tvíhlj", "tvt", "till", "to", "umr", "uh", "us", "uppl", "útg", "vb", "Vf", "vh", "vkf",
        "Vl", "vl", "lf", "vmf", "8vo", "vsk", "vth", "þt", "þf", "þjs", "þgf", "þlt", "þolm", "þm", "þml", "þýð"),
      numericPrefixes = Set("no", "No", "nr", "Nr", "nR", "NR")),
    irish -> NonBreakingPrefixes(
      prefixes = Set(
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
        "W", "X", "Y", "Z", "Á", "É", "Í", "Ó", "Ú", "", "Uacht", "Dr", "B.Arch", "", "m.sh", ".i", "Co", "Cf", "cf",
        "i.e", "r", "Chr"),
      numericPrefixes = Set("lch", "lgh", "uimh")
    ),
    italian -> NonBreakingPrefixes(
      prefixes = Set(
        // Any single upper case letter followed by a period is not at the end of a sentence (excluding "I"
        // occasionally, but we leave it in) usually upper case letters are initials in a name.
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
        "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        // List of titles. These are often followed by upper-case names, but do not indicate sentence breaks.
        "Adj", "Adm", "Adv", "Amn ", "Arch ", "Asst", "Avv", "Bart", "Bcc", "Bldg", "Brig", "Bros", "C.A.P", "C.P",
        "Capt", "Cc", "Cmdr", "Co", "Col", "Comdr", "Con", "Corp", "Cpl", "DR", "Dott", "Dr", "Drs", "Egr", "Ens",
        "Gen", "Geom", "Gov", "Hon", "Hosp", "Hr", "Id", "Ing", "Insp", "Lt", "MM", "MR", "MRS", "MS", "Maj", "Messrs",
        "Mlle", "Mme", "Mo", "Mons", "Mr", "Mrs", "Ms", "Msgr", "N.B", "Op", "Ord", "P.S", "P.T", "Pfc", "Ph", "Prof",
        "Pvt", "RP", "RSVP", "Rag", "Rep", "Reps", "Res", "Rev", "Rif", "Rt", "S.A", "S.B.F", "S.P.M", "S.p.A", "S.r.l",
        "Sen", "Sens", "Sfc", "Sgt", "Sig", "Sigg", "Soc", "Spett", "Sr", "St", "Supt", "Surg", "V.P", "", "# other",
        "a.c", "acc", "all ", "banc", "c.a", "c.c.p", "c.m", "c.p", "c.s", "c.v", "corr", "dott", "e.p.c", "ecc", "es",
        "fatt", "gg", "int", "lett", "ogg", "on", "p.c", "p.c.c", "p.es", "p.f", "p.r", "p.v", "post", "pp", "racc",
        "ric", "s.n.c", "seg", "sgg", "ss", "tel", "u.s", "v.r", "v.s",
        // Miscellaneous symbols - we add period-ending items that never indicate breaks ("p.m." does not fall into this
        // category - it sometimes ends a sentence).
        "v", "vs", "i.e", "rev", "e.g", "Nos", "Nr"),
      numericPrefixes = Set(
        // Numbers only. These should only induce breaks when followed by a numeric sequence.
        "No", "Art", "pp")),
    latvian -> NonBreakingPrefixes(
      prefixes = Set(
        // Any single upper case letter followed by a period is not at the end of a sentence (excluding "I"
        // occasionally, but we leave it in) usually upper case letters are initials in a name.
        "A", "Ā", "B", "C", "Č", "D", "E", "Ē", "F", "G", "Ģ", "H", "I", "Ī", "J", "K", "Ķ", "L", "Ļ",
        "M", "N", "Ņ", "O", "P", "Q", "R", "S", "Š", "T", "U", "Ū", "V", "W", "X", "Y", "Z", "Ž",
        // List of titles. These are often followed by upper-case names, but do not indicate sentence breaks.
        "dr", "Dr", "med", "prof", "Prof", "inž", "Inž", "ist.loc", "Ist.loc", "kor.loc",
        "Kor.loc", "v.i", "vietn", "Vietn",
        // Miscellaneous symbols.
        "a.l", "t.p", "pārb", "Pārb", "vec", "Vec", "inv", "Inv", "sk", "Sk", "spec", "Spec", "vienk", "Vienk", "virz",
        "Virz", "māksl", "Māksl", "mūz", "Mūz", "akad", "Akad", "soc", "Soc", "galv", "Galv", "vad", "Vad", "sertif",
        "Sertif", "folkl", "Folkl", "hum", "Hum"),
      numericPrefixes = Set("Nr")),
    lithuanian -> NonBreakingPrefixes(
      prefixes = Set(
        // Any single upper case letter followed by a period is not at the end of a sentence (excluding "I"
        // occasionally, but we leave it in) usually upper case letters are initials in a name.
        "A", "Ā", "B", "C", "Č", "D", "E", "Ē", "F", "G", "Ģ", "H", "I", "Ī", "J", "K", "Ķ", "L", "Ļ",
        "M", "N", "Ņ", "O", "P", "Q", "R", "S", "Š", "T", "U", "Ū", "V", "W", "X", "Y", "Z", "Ž",
        // Other symbols.
        "Dz", "Dž", "Just", "m", "mėn", "d", "g", "gim", "Pr", "Pn", "Pirm", "Antr", "Treč", "Ketv", "Penkt", "Šešt",
        "Sekm", "Saus", "Vas", "Kov", "Bal", "Geg", "Birž", "Liep", "Rugpj", "Rugs", "Spal", "Lapkr", "Gruod",
        "a", "adv", "akad", "aklg", "akt", "al", "A.V", "aps", "apskr", "apyg", "aps", "apskr", "asist", "asmv", "avd",
        "a.k", "asm", "asm.k", "atsak", "atsisk", "sąsk", "aut", "b", "k", "b.k", "bkl", "bt", "buv", "dail", "dek",
        "dėst", "dir", "dirig", "doc", "drp", "dš", "egz", "eil", "ekon", "el", "etc", "ež", "faks", "fak", "gen",
        "gyd", "gv", "įl", "Įn", "insp", "pan", "t.t", "k.a", "kand", "kat", "kyš", "kl", "kln", "kn", "koresp", "kpt",
        "kr", "kt", "kun", "l", "e", "p", "l.e.p", "ltn", "m", "mst", "m.e", "m.m", "mot", "mstl", "mgr", "mgnt", "mjr",
        "mln", "mlrd", "mok", "mokyt", "moksl", "nkt", "ntk", "Nr", "nr", "p", "p.d", "a.d", "p.m.e", "pan", "pav",
        "pavad", "pirm", "pl", "plg", "plk", "pr", "Kr", "pr.Kr", "prok", "prot", "pss", "pšt", "pvz", "r", "red", "rš",
        "sąs", "saviv", "sav", "sekr", "sen", "sk", "skg", "skyr", "sk", "skv", "sp", "spec", "sr", "st", "str", "stud",
        "š", "š.m", "šnek", "tir", "tūkst", "up", "upl", "vad", "vlsč", "ved", "vet", "virš", "vyr", "vyresn", "vlsč",
        "vs", "Vt", "vt", "vtv", "vv", "žml", "air", "amer", "anat", "angl", "arab", "archeol", "archit", "asm", "astr",
        "austral", "aut", "av", "bažn", "bdv", "bibl", "biol", "bot", "brt", "brus", "buh", "chem", "col", "con",
        "conj", "dab", "dgs", "dial", "dipl", "dktv", "džn", "ekon", "el", "esam", "euf", "fam", "farm", "filol",
        "filos", "fin", "fiz", "fiziol", "flk", "fon", "fot", "geod", "geogr", "geol", "geom", "glžk", "gr", "gram",
        "her", "hidr", "ind", "iron", "isp", "ist", "istor", "it", "įv", "reikšm", "įv.reikšm", "jap", "juok", "jūr",
        "kalb", "kar", "kas", "kin", "klaus", "knyg", "kom", "komp", "kosm", "kt", "kul", "kuop", "l", "lit", "lingv",
        "log", "lot", "mat", "maž", "med", "medž", "men", "menk", "metal", "meteor", "min", "mit", "mok", "ms", "muz",
        "n", "neig", "neol", "niek", "ofic", "opt", "orig", "p", "pan", "parl", "pat", "paž", "plg", "poet", "poez",
        "poligr", "polit", "ppr", "pranc", "pr", "priet", "prek", "prk", "prs", "psn", "psich", "pvz", "r", "rad",
        "rel", "ret", "rus", "sen", "sl", "sov", "spec", "sport", "stat", "sudurt", "sutr", "suv", "š", "šach", "šiaur",
        "škot", "šnek", "teatr", "tech", "techn", "teig", "teis", "tekst", "tel", "teol", "v", "t.p", "t", "p", "t.t",
        "t.y", "vaik", "vart", "vet", "vid", "vksm", "vns", "vok", "vulg", "zool", "žr", "ž.ū", "ž", "ū", "Em.", "Gerb",
        "gerb", "malon", "Prof", "prof", "Dr", "dr", "habil", "med", "inž", "Inž")),
    polish -> NonBreakingPrefixes(
      prefixes = Set(
        "adw", "afr", "akad", "al", "Al", "am", "amer", "arch", "art", "Art", "artyst", "astr", "austr", "bałt", "bdb",
        "bł", "bm", "br", "bryg", "bryt", "centr", "ces", "chem", "chiń", "chir", "c.k", "c.o", "cyg", "cyw", "cyt",
        "czes", "czw", "cd", "Cd", "czyt", "ćw", "ćwicz", "daw", "dcn", "dekl", "demokr", "det", "diec", "dł", "dn",
        "dot", "dol", "dop", "dost", "dosł", "h.c", "ds", "dst", "duszp", "dypl", "egz", "ekol", "ekon", "elektr", "em",
        "ew", "fab", "farm", "fot", "fr", "gat", "gastr", "geogr", "geol", "gimn", "głęb", "gm", "godz", "górn", "gosp",
        "gr", "gram", "hist", "hiszp", "hr", "Hr", "hot", "id", "in", "im", "iron", "jn", "kard", "kat", "katol", "k.k",
        "kk", "kol", "kl", "k.p.a", "kpc", "k.p.c", "kpt", "kr", "k.r", "krak", "k.r.o", "kryt", "kult", "laic", "łac",
        "niem", "woj", "nb", "np", "Nb", "Np", "pol", "pow", "m.in", "pt", "ps", "Pt", "Ps", "cdn", "jw", "ryc", "rys",
        "Ryc", "Rys", "tj", "tzw", "Tzw", "tzn", "zob", "ang", "ub", "ul", "pw", "pn", "pl", "al", "k", "n", "ww", "wł",
        "ur", "zm", "żyd", "żarg", "żyw", "wył", "bp", "bp", "wyst", "tow", "Tow", "o", "sp", "Sp", "st", "spółdz",
        "Spółdz", "społ", "spółgł", "stoł", "stow", "Stoł", "Stow", "zn", "zew", "zewn", "zdr", "zazw", "zast", "zaw",
        "zał", "zal", "zam", "zak", "zakł", "zagr", "zach", "adw", "Adw", "lek", "Lek", "med", "mec", "Mec", "doc",
        "Doc", "dyw", "dyr", "Dyw", "Dyr", "inż", "Inż", "mgr", "Mgr", "dh", "dr", "Dh", "Dr", "p", "P", "red", "Red",
        "prof", "prok", "Prof", "Prok", "hab", "płk", "Płk", "nadkom", "Nadkom", "podkom", "Podkom", "ks", "Ks", "gen",
        "Gen", "por", "Por", "reż", "Reż", "przyp", "Przyp", "śp", "św", "śW", "Śp", "Św", "ŚW", "szer", "Szer", "tel",
        "poz", "pok", "oo", "oO", "Oo", "OO", "najśw", "Najśw", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
        "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Ś", "Ć", "Ż", "Ź", "Dz"),
      numericPrefixes = Set("nr", "Nr", "pkt", "str", "tab", "Tab", "ust", "par", "r", "l", "s")),
    portuguese -> NonBreakingPrefixes(
      prefixes = Set(
        // Any single upper case letter followed by a period is not at the end of a sentence (excluding "I"
        // occasionally, but we leave it in) usually upper case letters are initials in a name.
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
        "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
        "s", "t", "u", "v", "w", "x", "y", "z",
        // Roman Numerals. A dot after one of these is not a sentence break in Portuguese.
        "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII",
        "XVIII", "XIX", "XX", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv",
        "xv", "xvi", "xvii", "xviii", "xix", "xx",
        // List of titles. These are often followed by upper-case names, but do not indicate sentence breaks.
        "Adj", "Adm", "Adv", "Art", "Ca", "Capt", "Cmdr", "Col", "Comdr", "Con", "Corp", "Cpl", "DR", "DRA", "Dr",
        "Dra", "Dras", "Drs", "Eng", "Enga", "Engas", "Engos", "Ex", "Exo", "Exmo", "Fig", "Gen", "Hosp", "Insp", "Lda",
        "MM", "MR", "MRS", "MS", "Maj", "Mrs", "Ms", "Msgr", "Op", "Ord", "Pfc", "Ph", "Prof", "Pvt", "Rep", "Reps",
        "Res", "Rev", "Rt", "Sen", "Sens", "Sfc", "Sgt", "Sr", "Sra", "Sras", "Srs", "Sto", "Supt", "Surg", "adj",
        "adm", "adv", "art", "cit", "col", "con", "corp", "cpl", "dr", "dra", "dras", "drs", "eng", "enga", "engas",
        "engos", "ex", "exo", "exmo", "fig", "op", "prof", "sr", "sra", "sras", "srs", "sto",
        // Miscellaneous symbols.
        "v", "vs", "i.e", "rev", "e.g", "Nos", "Nr"),
      numericPrefixes = Set(
        // Numbers only. These should only induce breaks when followed by a numeric sequence.
        "No", "Art", "pp")),
    romanian -> NonBreakingPrefixes(
      prefixes = Set(
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
        "W", "X", "Y", "Z", "dpdv", "etc", "șamd", "M.Ap.N", "dl", "Dl", "d-na", "D-na", "dvs", "Dvs", "pt", "Pt")),
    russian -> NonBreakingPrefixes(
      prefixes = Set(
        "А", "Б", "В", "Г", "Д", "Е", "Ж", "З", "И", "Й", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "У", "Ф", "Х",
        "Ц", "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь", "Э", "Ю", "Я", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
        "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0гг", "1гг", "2гг", "3гг", "4гг", "5гг",
        "6гг", "7гг", "8гг", "9гг", "0г", "1г", "2г", "3г", "4г", "5г", "6г", "7г", "8г", "9г", "Xвв", "Vвв", "Iвв",
        "Lвв", "Mвв", "Cвв", "Xв", "Vв", "Iв", "Lв", "Mв", "Cв", "0м", "1м", "2м", "3м", "4м", "5м", "6м", "7м", "8м",
        "9м", "0мм", "1мм", "2мм", "3мм", "4мм", "5мм", "6мм", "7мм", "8мм", "9мм", "0см", "1см", "2см", "3см", "4см",
        "5см", "6см", "7см", "8см", "9см", "0дм", "1дм", "2дм", "3дм", "4дм", "5дм", "6дм", "7дм", "8дм", "9дм", "0л",
        "1л", "2л", "3л", "4л", "5л", "6л", "7л", "8л", "9л", "0км", "1км", "2км", "3км", "4км", "5км", "6км", "7км",
        "8км", "9км", "0га", "1га", "2га", "3га", "4га", "5га", "6га", "7га", "8га", "9га", "0кг", "1кг", "2кг", "3кг",
        "4кг", "5кг", "6кг", "7кг", "8кг", "9кг", "0т", "1т", "2т", "3т", "4т", "5т", "6т", "7т", "8т", "9т", "0г",
        "1г", "2г", "3г", "4г", "5г", "6г", "7г", "8г", "9г", "0мг", "1мг", "2мг", "3мг", "4мг", "5мг", "6мг", "7мг",
        "8мг", "9мг", "бульв", "в", "вв", "г", "га", "гг", "гл", "гос", "д", "дм", "доп", "др", "е", "ед", "ед", "зам",
        "и", "инд", "исп", "Исп", "к", "кап", "кг", "кв", "кл", "км", "кол", "комн", "коп", "куб", "л", "лиц", "лл",
        "м", "макс", "мг", "мин", "мл", "млн", "млрд", "мм", "н", "наб", "нач", "неуд", "ном", "о", "обл", "обр", "общ",
        "ок", "ост", "отл", "п", "пер", "перераб", "пл", "пос", "пр", "просп", "проф", "р", "ред", "руб", "с", "сб",
        "св", "см", "соч", "ср", "ст", "стр", "т", "тел", "Тел", "тех", "тт", "туп", "тыс", "уд", "ул", "уч", "физ",
        "х", "хор", "ч", "чел", "шт", "экз", "э")),
    slovak -> NonBreakingPrefixes(
      prefixes = Set(
        "Bc", "Mgr", "RNDr", "PharmDr", "PhDr", "JUDr", "PaedDr", "ThDr", "Ing", "MUDr", "MDDr", "MVDr", "Dr", "ThLic",
        "PhD", "ArtD", "ThDr", "Dr", "DrSc", "CSs", "prof", "obr", "Obr", "Č", "č", "absol", "adj", "admin", "adr",
        "Adr", "adv", "advok", "afr", "ak", "akad", "akc", "akuz", "et", "al", "alch", "amer", "anat", "angl", "Angl",
        "anglosas", "anorg", "ap", "apod", "arch", "archeol", "archit", "arg", "art", "astr", "astrol", "astron", "atp",
        "atď", "austr", "Austr", "aut", "belg", "Belg", "bibl", "Bibl", "biol", "bot", "bud", "bás", "býv", "cest",
        "chem", "cirk", "csl", "čs", "Čs", "dat", "dep", "det", "dial", "diaľ", "dipl", "distrib", "dokl", "dosl",
        "dopr", "dram", "duš", "dv", "dvojčl", "dór", "ekol", "ekon", "el", "elektr", "elektrotech", "energet", "epic",
        "est", "etc", "etonym", "eufem", "európ", "Európ", "ev", "evid", "expr", "fa", "fam", "farm", "fem", "feud",
        "fil", "filat", "filoz", "fi", "fon", "form", "fot", "fr", "Fr", "franc", "Franc", "fraz", "fut", "fyz",
        "fyziol", "garb", "gen", "genet", "genpor", "geod", "geogr", "geol", "geom", "germ", "gr", "Gr", "gréc", "Gréc",
        "gréckokat", "hebr", "herald", "hist", "hlav", "hosp", "hromad", "hud", "hypok", "ident", "i.e", "ident", "imp",
        "impf", "indoeur", "inf", "inform", "instr", "int", "interj", "inšt", "inštr", "iron", "jap", "Jap", "jaz",
        "jedn", "juhoamer", "juhových", "juhozáp", "juž", "kanad", "Kanad", "kanc", "kapit", "kpt", "kart", "katastr",
        "knih", "kniž", "komp", "konj", "konkr", "kozmet", "krajč", "kresť", "kt", "kuch", "lat", "latinskoamer", "lek",
        "lex", "lingv", "lit", "litur", "log", "lok", "max", "Max", "maď", "Maď", "medzinár", "mest", "metr", "mil",
        "Mil", "min", "Min", "miner", "ml", "mld", "mn", "mod", "mytol", "napr", "nar", "Nar", "nasl", "nedok", "neg",
        "negat", "neklas", "nem", "Nem", "neodb", "neos", "neskl", "nesklon", "nespis", "nespráv", "neved", "než",
        "niekt", "niž", "nom", "náb", "nákl", "námor", "nár", "obch", "obj", "obv", "obyč", "obč", "občian", "odb",
        "odd", "ods", "ojed", "okr", "Okr", "opt", "opyt", "org", "os", "osob", "ot", "ovoc", "par", "part", "pejor",
        "pers", "pf", "Pf ", "P.f", "p.f", "pl", "Plk", "pod", "podst", "pokl", "polit", "politol", "polygr", "pomn",
        "popl", "por", "porad", "porov", "posch", "potrav", "použ", "poz", "pozit", "poľ", "poľno", "poľnohosp",
        "poľov", "pošt", "pož", "prac", "predl", "pren", "prep", "preuk", "priezv", "Priezv", "privl", "prof", "práv",
        "príd", "príj", "prík", "príp", "prír", "prísl", "príslov", "príč", "psych", "publ", "pís", "písm", "pôv",
        "refl", "reg", "rep", "resp", "rozk", "rozlič", "rozpráv", "roč", "Roč", "ryb", "rádiotech", "rím", "samohl",
        "semest", "sev", "severoamer", "severových", "severozáp", "sg", "skr", "skup", "sl", "Sloven", "soc", "soch",
        "sociol", "sp", "spol", "Spol", "spoloč", "spoluhl", "správ", "spôs", "st", "star", "starogréc", "starorím",
        "s.r.o", "stol", "stor", "str", "stredoamer", "stredoškol", "subj", "subst", "superl", "sv", "sz", "súkr",
        "súp", "súvzť", "tal", "Tal", "tech", "tel", "Tel", "telef", "teles", "telev", "teol", "trans", "turist",
        "tuzem", "typogr", "tzn", "tzv", "ukaz", "ul", "Ul", "umel", "univ", "ust", "ved", "vedľ", "verb", "veter",
        "vin", "viď", "vl", "vod", "vodohosp", "pnl", "vulg", "vyj", "vys", "vysokoškol", "vzťaž", "vôb", "vých", "výd",
        "výrob", "výsk", "výsl", "výtv", "výtvar", "význ", "včel", "vš", "všeob", "zahr", "zar", "zariad", "zast",
        "zastar", "zastaráv", "zb", "zdravot", "združ", "zjemn", "zlat", "zn", "Zn", "zool", "zr", "zried", "zv",
        "záhr", "zák", "zákl", "zám", "záp", "západoeur", "zázn", "územ", "účt", "čast", "čes", "Čes", "čl", "čísl",
        "živ", "pr", "fak", "Kr", "p.n.l", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
        "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")),
    slovenian -> NonBreakingPrefixes(
      prefixes = Set(
        "dr", "Dr", "itd", "itn", "d", "jan", "Jan", "feb", "Feb", "mar", "Mar", "apr", "Apr", "jun", "Jun", "jul",
        "Jul", "avg", "Avg", "sept", "Sept", "sep", "Sep", "okt", "Okt", "nov", "Nov", "dec", "Dec", "tj", "Tj", "npr",
        "Npr", "sl", "Sl", "op", "Op", "gl", "Gl", "oz", "Oz", "prev", "dipl", "ing", "prim", "Prim", "cf", "Cf", "gl",
        "Gl", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
        "V", "W", "X", "Y", "Z"),
      numericPrefixes = Set("št", "Št")),
    swedish -> NonBreakingPrefixes(
      prefixes = Set(
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", 
        "W", "X", "Y", "Z", "AB", "G", "VG", "dvs", "etc", "from", "iaf", "jfr", "kl", "kr", "mao", "mfl", "mm", "osv",
        "pga", "tex", "tom", "vs")),
    spanish -> NonBreakingPrefixes(
      prefixes = Set(
        // Any single upper case letter followed by a period is not at the end of a sentence (excluding "I"
        // occasionally, but we leave it in) usually upper case letters are initials in a name.
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
        "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        // Period-final abbreviation list from http://www.ctspanish.com/words/abbreviations.htm.
        "A.C", "Apdo", "Av", "Bco", "CC.AA", "Da", "Dep", "Dn", "Dr", "Dra", "EE.UU", "Excmo", "FF.CC", "Fil ", "Gral",
        "J.C", "Let", "Lic", "N.B", "P.D", "PV.P", "Prof", "Pts", "Rte", "S.A", "S.A.R", "S.E", "S.L", "S.R.C", "Sr",
        "Sra", "Srta", "Sta", "Sto", "T.V.E", "Tel", "Ud", "Uds", "V.B", "V.E", "Vd", "Vds", "a/c", "adj", "admón",
        "afmo", "apdo", "av", "c", "c.f", "c.g", "cap", "cm", "cta", "dcha", "doc", "ej", "entlo", "esq", "etc", "f.c",
        "gr ", "grs", "izq", "kg", "km", "mg", "mm", "nÃºm", "núm", "p", "p.a", "p.ej", "ptas", "pÃ¡g ", "pÃ¡gs", "pág",
        "págs", "q.e.g.e", "q.e.s.m", "s", "s.s.s", "vid", "vol")
    ),
    tamil -> NonBreakingPrefixes(
      prefixes = Set(
        "அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ", "ஐ", "ஒ", "ஓ", "ஔ", "ஃ", "க", "கா", "கி", "கீ", "கு", "கூ", "கெ",
        "கே", "கை", "கொ", "கோ", "கௌ", "க்", "ச", "சா", "சி", "சீ", "சு", "சூ", "செ", "சே", "சை", "சொ", "சோ",
        "சௌ", "ச்", "ட", "டா", "டி", "டீ", "டு", "டூ", "டெ", "டே", "டை", "டொ", "டோ", "டௌ", "ட்", "த", "தா", "தி",
        "தீ", "து", "தூ", "தெ", "தே", "தை", "தொ", "தோ", "தௌ", "த்", "ப", "பா", "பி", "பீ", "பு", "பூ", "பெ", "பே",
        "பை", "பொ", "போ", "பௌ", "ப்", "ற", "றா", "றி", "றீ", "று", "றூ", "றெ", "றே", "றை", "றொ", "றோ", "றௌ", "ற்",
        "ய", "யா", "யி", "யீ", "யு", "யூ", "யெ", "யே", "யை", "யொ", "யோ", "யௌ", "ய்", "ர", "ரா", "ரி", "ரீ", "ரு", "ரூ",
        "ரெ", "ரே", "ரை", "ரொ", "ரோ", "ரௌ", "ர்", "ல", "லா", "லி", "லீ", "லு", "லூ", "லெ", "லே", "லை", "லொ",
        "லோ", "லௌ", "ல்", "வ", "வா", "வி", "வீ", "வு", "வூ", "வெ", "வே", "வை", "வொ", "வோ", "வௌ", "வ்", "ள",
        "ளா", "ளி", "ளீ", "ளு", "ளூ", "ளெ", "ளே", "ளை", "ளொ", "ளோ", "ளௌ", "ள்", "ழ", "ழா", "ழி", "ழீ", "ழு", "ழூ",
        "ழெ", "ழே", "ழை", "ழொ", "ழோ", "ழௌ", "ழ்", "ங", "ஙா", "ஙி", "ஙீ", "ஙு", "ஙூ", "ஙெ", "ஙே", "ஙை", "ஙொ",
        "ஙோ", "ஙௌ", "ங்", "ஞ", "ஞா", "ஞி", "ஞீ", "ஞு", "ஞூ", "ஞெ", "ஞே", "ஞை", "ஞொ", "ஞோ", "ஞௌ", "ஞ்",
        "ண", "ணா", "ணி", "ணீ", "ணு", "ணூ", "ணெ", "ணே", "ணை", "ணொ", "ணோ", "ணௌ", "ண்", "ந", "நா", "நி",
        "நீ", "நு", "நூ", "நெ", "நே", "நை", "நொ", "நோ", "நௌ", "ந்", "ம", "மா", "மி", "மீ", "மு", "மூ", "மெ", "மே",
        "மை", "மொ", "மோ", "மௌ", "ம்", "ன", "னா", "னி", "னீ", "னு", "னூ", "னெ", "னே", "னை", "னொ", "னோ",
        "னௌ", "ன்", "திரு", "திருமதி", "வண", "கௌரவ", "உ.ம்", "Nos", "Nr"),
      numericPrefixes = Set(
        // Numbers only. These should only induce breaks when followed by a numeric sequence.
        "No", "Art", "pp"))
  ).withDefaultValue(NonBreakingPrefixes())
}
