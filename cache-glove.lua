-- Parse command line arguments

cmd = torch.CmdLine()
cmd:text()
cmd:option('-hidden_size', 200, 'Hidden size of LSTM layer')
cmd:option('-glove_size', 100, 'Glove embedding size')
opt = cmd:parse(arg)

require 'data'
glove = require './glove'

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

torch.save('glove.t7', known)
