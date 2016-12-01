function map(list, fn)
    local mapped = {}
    for i = 1, #list do
        mapped[i] = fn(list[i])
    end
    return mapped
end

function trim(s) return s:gsub('^%s+', ''):gsub('%s+$', '') end

function mapObject(obj, fn)
    local mapped = {}
    for k, v in pairs(obj) do
        mapped[k] = fn(v)
    end
    return mapped
end

function zip(l1, l2)
    local l = {}
    for i = 1, #l1 do
        l[i] = {l1[i], l2[i]}
    end
    return l
end

function filter(list, fn)
    local filtered = {}
    for i = 1, #list do
        if fn(list[i]) then
            table.insert(filtered, list[i])
        end
    end
    return filtered
end

function concat(t1, t2)
    local concatted = {}
    for _, i1 in pairs(t1) do
        table.insert(concatted, i1)
    end
    for _, i2 in pairs(t2) do
        table.insert(concatted, i2)
    end
    return concatted
end

function iteratorToTable(it)
    local t = {}
    for i in it do
        table.insert(t, i)
    end
    return t
end

function shuffleTable(t)
    local iterations = #t
    local j

    for i = iterations, 2, -1 do
        j = math.random(i)
        t[i], t[j] = t[j], t[i]
    end
end

function slugify(s)
    return s:lower():gsub("%W+", "_")
end

function keys(l)
    local ks = {}
    for k, v in pairs(l) do
        table.insert(ks, k)
    end
    return ks
end

function randomChoice(l)
    local i = math.ceil(math.random() * #l)
    return l[i], i
end

function randomKey(l)
    return randomChoice(keys(l))
end

function randomSample(l, n)
    local ss = {}
    for i = 1, n do
        ss[i] = randomChoice(l)
    end
    return ss
end

function tensorToString(t)
    local s = '{ '
    for i = 1, t:size(1) do
        s = s .. t[i]
        s = s .. ' '
    end
    s = s .. '}'
    return s
end

word_re = '[~%$%w_%.]+'
token_re = '%$[%w_%.]+'
syn_re = '~%w+'

function tokenize(s)
    return iteratorToTable(s:gmatch(word_re))
end

function countWords(s)
    return #iteratorToTable(s:gmatch(word_re))
end
