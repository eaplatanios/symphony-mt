import org.platanios.symphony.mt.Language._
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.data.loaders.IWSLT15Loader
import org.platanios.symphony.mt.data.processors._
import org.platanios.symphony.mt.vocabulary._

import java.nio.file.{Path, Paths}

object LoadPairwiseDataset {

  // #load_pairwise_dataset_example
  val dataConfig = DataConfig(
    // Loader
    workingDir = Paths.get("data"),
    loaderBufferSize = 8192,
    tokenizer = MosesTokenizer(),
    cleaner = MosesCleaner(),
    vocabulary = GeneratedVocabulary(
      SimpleVocabularyGenerator(sizeThreshold = 50000, countThreshold = -1)),
    // Corpus
    parallelPortion = 1.0f,
    trainBatchSize = 128,
    inferBatchSize = 32,
    evaluateBatchSize = 32,
    numBuckets = 5,
    srcMaxLength = 50,
    tgtMaxLength = 50,
    bufferSize = -1L,
    numShards = 1,
    shardIndex = 0,
    numParallelCalls = 4,
    // Vocabulary
    unknownToken = Vocabulary.UNKNOWN_TOKEN,
    beginOfSequenceToken = Vocabulary.BEGIN_OF_SEQUENCE_TOKEN,
    endOfSequenceToken = Vocabulary.END_OF_SEQUENCE_TOKEN)

  val loader = IWSLT15Loader(English, Vietnamese, dataConfig)
  // #load_pairwise_dataset_example
}
