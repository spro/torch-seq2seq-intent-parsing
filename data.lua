print(string.rep('-', 80))
math.randomseed(os.clock())
require 'helpers'

USE_GLOVE = true
glove = torch.load('glove.t7')

require './bash-templates'

function isArgument(arg)
    return arg:match('^%$')
end

avi = 1
argument_value_to_index = {}
argument_index_to_value = {}

-- Flatten into a single list of indexes

function maybeAddArgumentValue(argument_value)
    if argument_value_to_index[argument_value] == nil then
        argument_value_to_index[argument_value] = avi
        argument_index_to_value[avi] = argument_value
        avi = avi + 1
    end
end

for mk, mv in pairs(argument_values) do
    for tk, tv in pairs(mv) do
        maybeAddArgumentValue('$' .. mk .. '.' .. tk)
        for vk, vv in pairs(tv) do
            maybeAddArgumentValue(vk)
            print('at vk', vk)
        end
    end
end

-- Add static words from command templates

for _, sentence_command_template in pairs(sentence_command_templates) do
    for _, command_word in pairs(sentence_command_template[2]) do
        maybeAddArgumentValue(command_word)
    end
end

-- All words from input sentences

iwi = 1
input_word_to_index = {}
input_index_to_word = {}

function maybeAddInputWord(input_word)
    if input_word_to_index[input_word] == nil then
        input_word_to_index[input_word] = iwi
        input_index_to_word[iwi] = input_word
        iwi = iwi + 1
    end
end

function maybeAddInputWords(input_words)
    for _, input_word in pairs(tokenize(input_words)) do
        maybeAddInputWord(input_word)
    end
end

for _, sentence_command_template in pairs(sentence_command_templates) do
    for _, input_word in pairs(tokenize(sentence_command_template[1]:gsub("%$%w+%.%w+", ""))) do
        maybeAddInputWord(input_word)
    end
end

-- All words from noise pre and post

ns = {noise_pre, noise_post}
for _, n in pairs(ns) do
    for _, nv in pairs(n) do
        maybeAddInputWords(nv)
    end
end

-- All words from argument values

for mk, mv in pairs(argument_values) do
    for tk, tv in pairs(mv) do
        for vk, vv in pairs(tv) do
            for _, v in pairs(vv) do
                maybeAddInputWords(v)
            end
        end
    end
end

n_input_words = #keys(input_word_to_index)
UNK = n_input_words + 1
-- n_argument_names = #keys(argument_name_to_index)
n_argument_values = #keys(argument_value_to_index)
command_EOS = n_argument_values + 1

print('value to index', argument_value_to_index)
print('value to index', argument_value_to_index)

function makeSentence()
    local sentence_command_template = randomChoice(sentence_command_templates)
    local sentence_template, command_template = unpack(sentence_command_template)

    local method = command_template[1]

    local sentence = sentence_template

    -- Pull varible names out of sentence, 
    local argument_pairs = {}
    for argument_name in sentence:gfind('%$[%w.]+') do
        table.insert(argument_pairs, {argument_name})
    end

    -- Choose values to fill with
    for _, argument_pair in pairs(argument_pairs) do
        local argument_name = argument_pair[1]
        local _, _, argument_type = argument_name:find('%$(%w+)') -- e.g. "light"
        local _, _, argument_sub = argument_name:find('%$%w+.(%w+)') -- e.g. "device"
        local argument_slug = randomKey(argument_values[argument_type][argument_sub])
        local argument_value = randomChoice(argument_values[argument_type][argument_sub][argument_slug])

        argument_pair[2] = argument_slug
        argument_pair[3] = argument_value
    end

    -- Put them in the sentence
    for _, argument_pair in pairs(argument_pairs) do
        sentence = sentence:gsub(argument_pair[1], argument_pair[3])
    end

    if math.random() < 0.2 then
        sentence = randomChoice(noise_pre) .. ' ' .. sentence
    elseif math.random() < 0.2 then
        sentence = sentence .. ' ' .. randomChoice(noise_post)
    end

    -- Fill out sentence template for encoder inputs and decoder target
    local encoder_inputs = map(tokenize(sentence), function(encoder_input)
        return {
            glove[encoder_input],
            torch.LongTensor({input_word_to_index[encoder_input]})
        }
    end)

    -- Turn tokens in command template into indexes
    command_rendered = map(command_template, function(token)
        -- print('token', token)
        for _, argument_pair in pairs(argument_pairs) do
            -- print('pair', argument_pair)
            -- print('token', token)
            token = token:gsub(argument_pair[1], argument_pair[2])
            -- print('token now', token)
        end
        return token
    end)
    local command_indexes = map(command_rendered, function(argument_name)
        -- print('argument name is', argument_name)
        return argument_value_to_index[argument_name]
    end)
    -- print("indexes now", command_indexes)

    -- Command decoder inputs are EOS + indexes
    local command_decoder_inputs = concat({command_EOS}, command_indexes)
    command_decoder_inputs = asTensors(command_decoder_inputs)

    -- Command decoder targets are indexes + EOS
    local command_decoder_targets = concat(command_indexes, {command_EOS})

    -- Argument decoder inputs are each argument name from command as indexes
    local fillable_arguments = filter(command_template, isArgument)
    local argument_decoder_inputs = map(fillable_arguments, function(argument_name)
        return argument_value_to_index[argument_name]
    end)
    argument_decoder_inputs = asTensors(argument_decoder_inputs)

    -- Argument decoder targets are values per argument
    local argument_decoder_targets = {}
    for i, argument_pair in pairs(argument_pairs) do
        argument_decoder_targets[i] = argument_value_to_index[argument_pair[2]]
    end

    -- return sentence, command_template
    return sentence, encoder_inputs, command_decoder_inputs, command_decoder_targets, argument_decoder_inputs, argument_decoder_targets
end

print('makeSentence:', makeSentence())
