# Getting Started

The easiest way to start experimenting and playing around 
with the SMT codebase for research purposes is to clone the 
repository and start working with that. If you want to use 
SMT in production, the best way is to add a dependency to 
the published artifacts in your codebase. The following two 
sections describe each option.

TODO: Add installation instructions.

## Cloning the Repository

### Requirements

TODO:

After you have cloned the repository, the simplest way to 
reproduce the results from [our paper](TODO) and get going 
with running experiments is to first create a JAR that 
contains all dependencies using:

```bash
sbt assembly
cp target/scala-2.12/symphony-mt-0.1.0-SNAPSHOT.jar symphony-mt-0.1.0-SNAPSHOT.jar
```

Then, you can run experiments using commands like the 
following:

```bash
java -jar symphony-mt-0.1.0-SNAPSHOT.jar \
  --task train \
  --working-dir temp/experiments \
  --data-dir temp/data \
  --dataset iwslt17 \
  --languages de:en,de:it,de:ro,en:it,en:nl,en:ro,it:nl,nl:ro \
  --eval-languages de:en,de:it,de:nl,de:ro,en:de,en:it,en:nl,en:ro,it:de,it:en,it:nl,it:ro,nl:de,nl:it,nl:en,nl:ro,ro:de,ro:it,ro:nl,ro:en \
  --use-back-translations \
  --parallel-portion 1.00 \
  --eval-datasets dev2010:1.00,tst2017:1.00 \
  --eval-metrics bleu,hyp_len,ref_len,sen_cnt \
  --tokenizer moses \
  --cleaner moses \
  --vocabulary generated:20000:5 \
  --batch-size 128 \
  --num-buckets 5 \
  --src-max-length 50 \
  --tgt-max-length 50 \
  --buffer-size 128 \
  --model-arch bi_rnn:2:2 \
  --model-cell lstm:tanh \
  --model-type hyper_lang:8 \
  --word-embed-size 512 \
  --lang-embed-size 512 \
  --residual \
  --attention \
  --dropout 0.2 \
  --label-smoothing 0.1 \
  --beam-width 10 \
  --length-penalty 0.6 \
  --opt amsgrad:0.001 \
  --num-steps 1000000 \
  --summary-steps 100 \
  --checkpoint-steps 5000 \
  --log-loss-steps 100 \
  --log-eval-steps 5000 \
  --num-gpus 1 \
  --seed 10
```

The experiments command-line interface is described 
[here](experiments-cli.md).

## Adding as a Dependency

In order to add SMT as a dependency in your own project, 
you can use the following snippet:

@@dependency[sbt,Maven,Gradle] {
  group="org.platanios"
  artifact="symphony-mt"
  version="0.1.0"
}
