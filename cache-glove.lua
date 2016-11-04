opt = {glove_size=100}
require 'data'
glove = require '../torch/glove.torch/glove'

known = {}

for i = 1, n_input_words do
    word = input_index_to_word[i]
    print('word', word)
    if glove:word2vec(word) ~= nil then
        known[word] = glove:word2vec(word):clone():double()
    else
        known[word] = torch.zeros(opt.glove_size)
    end
end

common = {'up', 'down', 'hi', 'so', 'hmm'}

for i = 1, #common do
    word = common[i]
    print('word', word)
    known[word] = glove:word2vec(word):clone():double()
end

torch.save('glove.t7', known)
