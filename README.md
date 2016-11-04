# torch-seq2seq-intent-parser

"Intent parsing" reframed as a seq2seq translation problem from human to computer language.

The model is based off the standard seq2seq encoder-decoder model, using a single layer of GRU nodes for each side, plus attention over encoder hidden states in the decoder.

## Demo: en2bash

```
> find all python files in my projects folder
find ~/Projects -name *.py

> copy the nginx configuration to downloads
cp /var/nginx/conf/nginx.conf ~/Downloads

> find the oldest files in downloads
ls -lt ~/Downloads

> delete the root directory
rm -r /
```

## Demo: en2iot

```
> make the office light brighter
lights setState office_light up

> turn up that music
jukebox setState volume up

> how warm is it in the living room?
sensors getState living_room temperature

> make some tea
switches setState tea on
```