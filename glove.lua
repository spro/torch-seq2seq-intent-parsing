binfilename = './data/glove.twitter.27B.' .. opt.glove_size .. 'd.txt'
outfilename = './data/glove.twitter.27B.' .. opt.glove_size .. 'd.t7'

local GloVe = {}

if not paths.filep(outfilename) then
    print('Turning GloVe data to t7...')
    GloVe = require('./glove-to-t7.lua')
else
    print('Loading GloVe data from t7...')
    GloVe = torch.load(outfilename)
    print('Done reading GloVe data.')
end

GloVe.distance = function (self, vec, k)
    local k = k or 1    
    --self.zeros = self.zeros or torch.zeros(self.M:size(1));
    local norm = vec:norm(2)
    vec:div(norm)
    local distances = torch.mv(self.M ,vec)
    distances, oldindex = torch.sort(distances,1,true)
    local returnwords = {}
    local returndistances = {}
    for i = 1,k do
        table.insert(returnwords, self.v2wvocab[oldindex[i]])
        table.insert(returndistances, distances[i])
    end
    local top = {}
    for i, word in pairs(returnwords) do
        table.insert(top, {word, returndistances[i]})
    end
    return top
end

GloVe.word2vec = function (self, word, throwerror)
    local throwerror = throwerror or false
    local ind = self.w2vvocab[word]
    if throwerror then
        assert(ind ~= nil, 'Word does not exist in the dictionary!')
    end
    if ind == nil then
        ind = self.w2vvocab['UNK']
    end
    return self.M[ind]
end

return GloVe

