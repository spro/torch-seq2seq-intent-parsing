require 'nn'
require 'nngraph'
require 'optim'
display = require 'display'
require './plot'
require 'data'
require './model_utils'

models = torch.load('models.t7')
opt = torch.load('opt.t7')
glove = torch.load('glove.t7')

cmd = torch.CmdLine()
cmd:text()
cmd:option('-sentence', 'find leads named joe jones', 'Input sentence')
sample_opt = cmd:parse(arg)

clones = mapObject(models, function(model)
    return model_utils.clone_many_times(model, opt.max_length)
end)

print('models', models)
print('opt', opt)

function sample(sentence)
    -- Inputs and targets
    -- local sentence, encoder_inputs, command_decoder_inputs, _, _, _ = makeSentence()
    sentence_tokens = tokenize(sentence)
    print('--------\n[sample]', sentence)
    local encoder_inputs = map(iteratorToTable(sentence:gmatch(word_re)), function(word)
        return glove[word] or torch.zeros(opt.glove_size)
    end) 

	-- Forward pass
    -- -------------------------------------------------------------------------

    -- Forward through encoder
    local encoder_outputs = {[0] = torch.zeros(opt.hidden_size):double()}
    for t = 1, #encoder_inputs do
        encoder_outputs[t] = clones.encoder[t]:forward({encoder_inputs[t], encoder_outputs[t-1]})
    end

    -- Pad encoder outputs with 0s

    local encoder_outputs_padded = {}
    for t = 1, #encoder_outputs do
        encoder_outputs_padded[t] = encoder_outputs[t]
    end
    for t = 1, (opt.max_length - #encoder_outputs) do
        table.insert(encoder_outputs_padded, torch.zeros(opt.hidden_size))
    end

    last_encoder_output = encoder_outputs_padded[#encoder_inputs]

    -- Through command decoder
    local command_decoder_inputs = {[1] = torch.LongTensor({command_EOS})}
    local command_decoder_hidden_outputs = {[0] = last_encoder_output}
    local command_decoder_outputs = {}

    local sampled = ''
    local sampled_full = ''
    local sampled_tokens = {}

    for t = 1, opt.max_length do
        local command_decoder_output = clones.command_decoder[t]:forward(
            {command_decoder_inputs[t], command_decoder_hidden_outputs[t-1], encoder_outputs_padded})
        command_decoder_outputs[t] = command_decoder_output[1]
        command_decoder_hidden_outputs[t] = command_decoder_output[2][1]

        -- Choose most likely output
        out_max, out_max_index = command_decoder_outputs[t]:max(1)
        if out_max_index[1] == command_EOS then
            break
        end
        local output_argument_name = command_index_to_word[out_max_index[1]]
        table.insert(sampled_tokens, output_argument_name)
        sampled = sampled .. ' ' .. output_argument_name

        -- Next decoder input is current output
        command_decoder_inputs[t + 1] = out_max_index
    end

    -- Get arguments from command output

    local command_argument_ts = {}
    local command_argument_indexes = {}

    for t = 1, #command_decoder_outputs - 1 do
        local _, command_index = command_decoder_outputs[t]:max(1)
        command_index = command_index[1]
        local command_word = command_index_to_word[command_index]
        if command_word:match(token_re) then
            command_argument_ts[command_word] = t
            command_argument_indexes[command_word] = torch.LongTensor({command_index})
        end
        sampled_full = sampled_full .. ' ' .. command_word
    end

    -- For each argument output
    local command = sampled_full
    local args = {}

    for arg_name, arg_value in pairs(command_argument_indexes) do
        arg_t = command_argument_ts[arg_name]
        sampled_tokens = ''
        arg_out_s = '{'

        -- Forward through argument decoder

        local argument_decoder_hidden_outputs = {[0] = last_encoder_output}
        local argument_decoder_attention_outputs = {}
        local argument_decoder_outputs = {}
        local argument_decoder_output_indexes = {}

        for t = 1, #encoder_inputs do
            argument_decoder_outputs[t], argument_decoder_gru_outputs = unpack(clones.argument_decoder[t]:forward({
                command_decoder_hidden_outputs[arg_t],
                encoder_inputs[t],
                argument_decoder_hidden_outputs[t-1],
                encoder_outputs_padded
            }))
            argument_decoder_hidden_outputs[t], argument_decoder_attention_outputs[t] = unpack(argument_decoder_gru_outputs)
            if argument_decoder_outputs[t][1] > 0.5 then
                arg_out_s = arg_out_s .. ' ' .. '1'
                sampled_tokens = sampled_tokens .. sentence_tokens[t] .. ' '
            else
                arg_out_s = arg_out_s .. ' ' .. '0'
            end
        end

        args[arg_name:sub(2)] = trim(sampled_tokens)
        arg_out_s = arg_out_s .. ' }'
        sampled_full = sampled_full:gsub(arg_name, '( ' .. arg_name .. ' = ' .. sampled_tokens .. ')')
        print(arg_name, '->', arg_out_s)
    end

    print('sampled', sampled_full)
    return command, args
end

sample(sample_opt.sentence)

