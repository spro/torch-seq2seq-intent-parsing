function map(list, fn)
    local mapped = {}
    for i = 1, #list do
        mapped[i] = fn(list[i])
    end
    return mapped
end

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

function tokenize(s)
    local tokens = {}
    local s = s:lower():gsub("[@()/':;.,!\?-]", " %1 ")
    for i in string.gmatch(s, '%S+') do
        table.insert(tokens, i)
    end
    return tokens
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

function asTensors(l)
    return map(l, function (i) return torch.LongTensor({i}) end)
end

function slice(tbl, first, last, step)
    local sliced = {}

    for i = first or 1, last or #tbl, step or 1 do
        sliced[#sliced+1] = tbl[i]
    end

    return sliced
end

