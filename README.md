# Symphony Machine Translation

We plan to soon publish a documentation website with 
information on how to use this software package. For now, 
we provide some instructions on how to reproduce the 
experiments presented in our paper, soon to be presented 
at EMNLP 2018, *Contextual Parameter Generation for
Universal Neural Machine Translation*, Emmanouil A. 
Platanios, Mrinmaya Sachan, Graham Neubig, and Tom M. 
Mitchell.

## Running Experiments

In order to reproduce our experimental results you must
first execute `sbt assembly` on the root directory of this
repository, after cloning it.

Example experiment scripts are located in the `scripts` 
directory. We will soon update this with the specific 
scripts used to reproduce the experiments presented in 
our paper.

## Using Precompiled TensorFlow Distribution

In order to use the precompiled TensorFlow binaries that
TensorFlow Scala provides, you need to change line 76 in
the `build.sbt` file, from this:

```scala
libraryDependencies += "org.platanios" %% "tensorflow" % tensorFlowScalaVersion
```

to this:

```scala
libraryDependencies += "org.platanios" %% "tensorflow" % tensorFlowScalaVersion classifier "linux-cpu-x86_64"
```

*Make sure to replace `linux-cpu-x86_64` with the string
that corresponds to your platform.* Currently supported
platforms are: `linux-cpu-x86_64`, `linux-gpu-x86_64`, and
`darwin-cpu-x86_64`.

For more information on how to install/configure TensorFlow
Scala, please refer to the
[official website](http://platanios.org/tensorflow_scala/).

## TODOs

- [ ] Add support for separate source/target word
      embeddings.
- [ ] Add support for bridges between the encoder and
      the decoder (inspired from OpenNMT).
- [ ] Add support for merging the bidirectional encoder
      states using either summation or concatenation.
- [ ] Add support for other attention models, as in
      [here](https://arxiv.org/pdf/1508.04025.pdf).
- [ ] Add support for scheduled sampling, presented in
      [this paper](http://papers.nips.cc/paper/5956-scheduled-sampling-for-sequence-prediction-with-recurrent-neural-networks.pdf).
