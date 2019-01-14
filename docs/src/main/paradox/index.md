# Symphony Machine Translation

@@@ index

* [Getting Started](getting-started.md)
* [Data](data.md)
* [Models](models.md)
* [Learning](learning.md)
* [Experiments](experiments.md)

@@@

Symphony MT is a modular and extensible machine translation library that 
supports both pairwise and multilingual translation, across an arbitrary 
number of languages. It also supports zero-shot translation using either 
pivoting/bridging, or by providing models that natively support multilingual 
translation. It contains modules for:

  - Downloading and preprocessing data:
    - **Datasets:** IWSLT-14, IWSLT-15, IWSLT-16, IWSLT-17, WMT-16, etc.
    - **Processors:** data cleaning, normalization, tokenization, etc.
  - Constructing MT models using various features:
    - Various encoders and decoders (e.g., RNNs, bi-directional RNNs, and
      transformer).
    - Attention models (e.g., Bahdanau, Luong, and multi-head).
    - Greedy and beam search decoding.
  - Training MT models:
    - Lots of supported optimizers (e.g., SGD, Adam, AMSGrad, and YellowFin).
    - Checkpointing.
    - Monitoring (e.g., TensorBoard support).
    - Distributed (e.g., multi-GPU) training.
  - Evaluating MT models:
    - Can evaluate both while training and after.
    - Supports various metrics (e.g., BLEU, Rouge, Meteor, and TER).
  - Using trained MT models to perform inference.

TODO: Add paper and link.

This library was initially developed to support the proposed methods in 
[](), but currently supports a more diverse set of features than what was 
presented in that paper.

## Citation

If you use this library in your work, we would really appreciate it if 
you could cite the following paper:

TODO: Add citation information.

## License

Copyright 2017-2018, Emmanouil Antonios Platanios. All Rights Reserved. 
Symphony Machine Translation is provided under the Apache 2.0 license.
