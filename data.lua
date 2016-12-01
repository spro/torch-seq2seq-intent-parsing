require 'helpers'
require './templates'
glove = torch.load('glove.t7')

math.randomseed(os.time())

max_length = 20
NOISE_PROB = 0.3

function replaceSynonyms(s)
    return s:gsub(syn_re, function(word)
        return randomChoice(synonyms[word:sub(2)])
    end)
end

function makeSentence()
    local command = randomChoice(commands)
    local input_template = command[1]
    local output_template = command[2]
    local input = input_template
    local command_output = iteratorToTable(output_template:gmatch(word_re))

    -- Find tokens in input template (assume tokens are always at least in input)
    local tokens = {}
    for token in input_template:gmatch(token_re) do
        table.insert(tokens, token)
    end

    -- Choose values for tokens
    local token_values = {}
    for _, full_token in pairs(tokens) do
        token = full_token:sub(2)
        -- print('token', token, full_token)
        token_values[full_token] = randomChoice(freeform[argument_types[token] or token])
    end

    -- Add noise words pre and post input
    if math.random() < NOISE_PROB then
        local noise = randomChoice(freeform.noise_pre)
        input = noise .. ' ' .. input
    end
    if math.random() < NOISE_PROB then
        local noise = randomChoice(freeform.noise_post)
        input = input .. ' ' .. noise
    end

    -- Replace matching tokens in input
    -- Also replace any ~words with synonyms
    -- Also make token masks (array of binary values of which words are relevant)
    local token_masks = {}
    seen_words = 1
    input = input:gsub(word_re, function(word)
        local token_word = token_values[word]
        if token_word ~= nil then
            token_masks[word] = torch.zeros(max_length)
            token_word = replaceSynonyms(token_word)
            l = countWords(token_word)
            -- print('l', l)
            token_masks[word][{{seen_words, seen_words + l - 1}}] = 1
            seen_words = seen_words + l
            return token_word
        else
            word = replaceSynonyms(word)
            l = countWords(word)
            seen_words = seen_words + l
            return word
        end
    end)

    local encoder_inputs = map(iteratorToTable(input:gmatch(word_re)), function(word)
        return glove[word] or torch.zeros(opt.glove_size)
    end) 

    local command_indexes = map(command_output, function(word)
        return command_word_to_index[word]
    end)
    local command_decoder_inputs = concat({command_EOS}, command_indexes)
    local command_decoder_targets = concat(command_indexes, {command_EOS})
    command_decoder_inputs = map(command_decoder_inputs, function(input) return torch.LongTensor({input}) end)
    command_decoder_targets = map(command_decoder_targets, function(input) return torch.LongTensor({input}) end)

    return {input, encoder_inputs, command_decoder_inputs, command_decoder_targets, token_masks}
end

-- Count known input words (for caching glove vectors)

input_word_to_index = {}
input_index_to_word = {}
n_input_words = 0

function maybeAddInputWord(word)
    if input_word_to_index[word] == nil then
        n_input_words = n_input_words + 1
        input_word_to_index[word] = n_input_words
        input_index_to_word[n_input_words] = word
    end
end

for _, word_sets in pairs(freeform) do
    for _, words in pairs(word_sets) do
        for word in words:gmatch(word_re) do
            maybeAddInputWord(word)
        end
    end
end

for _, command in pairs(commands) do
    for word in command[1]:gmatch(word_re) do
        if word:match(token_re) == nil and word:match(syn_re) == nil then
            maybeAddInputWord(word)
        end
    end
end

for _, wordss in pairs(synonyms) do
    for _, words in pairs(wordss) do
        for word in words:gmatch(word_re) do
            maybeAddInputWord(word)
        end
    end
end

-- Count known command words (including argument placeholders)

command_word_to_index = {}
command_index_to_word = {}
n_command_words = 0

function maybeAddCommandWord(word)
    if command_word_to_index[word] == nil then
        n_command_words = n_command_words + 1
        command_word_to_index[word] = n_command_words
        command_index_to_word[n_command_words] = word
    end
end

for _, command in pairs(commands) do
    for word in command[2]:gmatch(word_re) do
        maybeAddCommandWord(word)
    end
end

command_EOS = n_command_words + 1

-- print(makeSentence())
