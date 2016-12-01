# intense

Intent parsing framed as human to computer language translation.

The model uses the seq2seq encoder-decoder model with two decoders: one to decode the intent and argument placeholders (`setState $light.name $light.state`), and one to align input words with slot values. Both are RNNs of GRU cells using an attention mechanism over the encoder outputs.

![](https://i.imgur.com/V1ltvhI.png)

Based on ideas from the following papers:

* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215v3)
* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473v7)
* [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393) adds a "copy mode" in addition to the standard decoder (combined as a mixture) which uses an attention-like mechanism to decide which words to copy from the input
* [Pointer Networks](https://arxiv.org/abs/1506.03134)

Sampling example showing input, slot alignment, and output:

```
[sample]        may you send Nenita Service a text
$person =>      { 0 0 0 1 1 0 0 }
(sampled)  sendText ( $person = Nenita Service )

[sample]        what is the cost of pesos
$market =>      { 0 0 0 0 0 1 }
(sampled)  market getPrice ( $market = pesos )

[sample]        could you send Deloise Kamps Readme.md please
$person =>      { 0 0 0 1 1 0 0 }
$file   =>      { 0 0 0 0 0 1 0 }
(sampled)  sendFile ( $person = Deloise Kamps ) ( $file = Readme.md )

[sample]        turn off my lights thanks
$light.state    =>      { 0 1 0 0 0 }
$light.name     =>      { 0 0 1 1 0 }
(sampled)  lights setState ( $light.name = my lights ) ( $light.state = off )
```

## Preparation

First download these 1.42 GB of 27 billion Twitter GloVe vectors from [http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip](http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip) and extract them to `data/glove.twitter.27B.*.txt`

Then run the `cache-glove` script to cache a subset of glove vectors (based on the sentence templates in templates.lua). This is to make it less painfully slow to start the training script (in case you want to quickly tweak a parameter and restart)

```bash
$ th cache-glove.lua
```

## Training

```
$ th train.lua
Usage: [options]

-hidden_size         Hidden size of LSTM layer [200]
-glove_size          Glove embedding size [100]
-dropout             Dropout [0.1]
-learning_rate       Learning rate [0.0002]
-learning_rate_decay Learning rate decay [1e-06]
-max_length          Maximum output length [20]
-n_epochs            Number of epochs to train [10000]
```

With the small example training set you should get good results around 2500 epochs (about 5 minutes on a recent MacBook CPU)

