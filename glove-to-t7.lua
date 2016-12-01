-- Reading Header

local encodingsize = -1
local ctr = 0

for line in io.lines(binfilename) do
    if ctr == 0 then
        for i in string.gmatch(line, "%S+") do
            encodingsize = encodingsize + 1
        end
    end
    ctr = ctr + 1
end

words = ctr
size = encodingsize
print('words', words, 'size', size)

local w2vvocab = {}
local v2wvocab = {}
local M = torch.Tensor(words, size)

-- Reading Contents

i = 1

for line in io.lines(binfilename) do
    xlua.progress(i, words)
    local vecrep = {}
    for ii in string.gmatch(line, "%S+") do
        table.insert(vecrep, ii)
    end

    str = vecrep[1]
    table.remove(vecrep, 1)

    if #vecrep == size then
        vecrep = torch.Tensor(vecrep)

        local norm = torch.norm(vecrep,2)
        if norm ~= 0 then vecrep:div(norm) end
        w2vvocab[str] = i
        v2wvocab[i] = str
        M[{{i},{}}] = vecrep
        i = i + 1
    end
end

-- Writing Files
GloVe = {}
GloVe.M = M
GloVe.w2vvocab = w2vvocab
GloVe.v2wvocab = v2wvocab
torch.save(outfilename,GloVe)
print('Writing t7 File for future usage.')

return GloVe

