import org.platanios.symphony.mt.Language._
import org.platanios.symphony.mt.data._
import org.platanios.symphony.mt.data.loaders.IWSLT15Loader
import org.platanios.symphony.mt.data.processors._
import org.platanios.symphony.mt.vocabulary._

import java.nio.file.Paths

object LoadPairwiseDataset {

  // #load_pairwise_dataset_example
  val dataConfig = DataConfig(
    // Loader
    dataDir = Paths.get("data"),
    loaderBufferSize = 8192,
    tokenizer = MosesTokenizer(),
    cleaner = MosesCleaner(),
    vocabulary = GeneratedVocabulary(SimpleVocabularyGenerator(sizeThreshold = 50000, countThreshold = -1)),
    // Corpus
    trainBatchSize = 128,
    inferBatchSize = 32,
    evalBatchSize = 32,
    numBuckets = 5,
    srcMaxLength = 50,
    tgtMaxLength = 50,
    shuffleBufferSize = -1L,
    numParallelCalls = 4)

  val loader = IWSLT15Loader(English, Vietnamese, dataConfig)
  // #load_pairwise_dataset_example
}
