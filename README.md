# intense

"Intent parsing" reframed as a seq2seq translation problem from human to computer language.

The model uses the seq2seq encoder-decoder model with two decoders: one to decode the intent, and one to decode slots. Both use attention mechanisms over the encoder outputs.

Based on ideas from the following papers:

* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215v3)
* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473v7)
* [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393) adds a "copy mode" in addition to the standard decoder (combined as a mixture) which uses an attention-like mechanism to decide which words to copy from the input
* [Pointer Networks](https://arxiv.org/abs/1506.03134)

```
[sample]        may you send Nenita Service a text
$person =>      { 0 0 0 1 1 0 0 }
sampled  sendText ( $person = Nenita Service )

[sample]        what is the cost of pesos
$market =>      { 0 0 0 0 0 1 }
sampled  market getPrice ( $market = pesos )

[sample]        could you send Deloise Kamps Readme.md please
$person =>      { 0 0 0 1 1 0 0 }
$file   =>      { 0 0 0 0 0 1 0 }
sampled  sendFile ( $person = Deloise Kamps ) ( $file = Readme.md )

[sample]        turn off my lights thanks
$light.state    =>      { 0 1 0 0 0 }
$light.name     =>      { 0 0 1 1 0 }
sampled  lights setState ( $light.name = my lights ) ( $light.state = off )
```

## Usage

### Training

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

