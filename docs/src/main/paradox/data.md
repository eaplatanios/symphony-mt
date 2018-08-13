# Data

The simplest place to start for learning to work with the 
data pipeline is an example for how to load a parallel 
dataset between two languages:

@@snip [LoadPairwiseDataset.scala](../scala/LoadPairwiseDataset.scala) { #load_pairwise_dataset_example }

## Downloading

Depending on the data loader you choose to use 
(e.g., `IWSLT15Loader` in the example above), and the 
languages for which you request for data to be loaded, 
Symphony Machine Translation will download the necessary 
files on the working directory that you specify. It will 
use a buffer with size `loaderBufferSize` while 
downloading data.

## Preprocessing
