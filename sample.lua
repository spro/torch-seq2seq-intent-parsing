require 'nn'
require 'nngraph'
require 'optim'
require 'helpers'
require 'data'
display = require 'display'

-- Parse command line arguments

cmd = torch.CmdLine()
cmd:text()
cmd:option('-sentence', 'turn on the office light', 'Sentence to translate')

sample_opt = cmd:parse(arg)

-- Load saved info

glove = torch.load('glove.t7')
opt = torch.load('opt.t7')
models = torch.load('models.t7')

function sample(input_string)

    print('\n[sample]', input_string)
    encoder_inputs = map(tokenize(input_string), function (word)
        return {
            glove[word] or torch.zeros(opt.hidden_size),
            torch.LongTensor({input_word_to_index[word] or UNK})
        }
    end)

	-- Forward pass
    -- -------------------------------------------------------------------------

    -- Through encoder
    local encoder_outputs = {[0] = torch.zeros(opt.hidden_size):double()}
    for t = 1, #encoder_inputs do
        encoder_outputs[t] = models.encoder:forward({encoder_inputs[t][1], encoder_inputs[t][2], encoder_outputs[t-1]})
    end
    local last_encoder_output = encoder_outputs[#encoder_inputs]

    -- Pad encoder outputs with 0s

    local encoder_outputs_padded = {}
    for t = 1, #encoder_outputs do
        -- table.insert(encoder_outputs_padded, encoder_outputs[i])
        table.insert(encoder_outputs_padded, encoder_outputs[t])
    end
    for t = 1, (opt.max_length - #encoder_outputs) do
        table.insert(encoder_outputs_padded, torch.zeros(opt.hidden_size))
    end

    -- Through command decoder
    local command_decoder_inputs = {[1] = torch.LongTensor({n_argument_values + 1})}
    local command_decoder_in_outputs = {}
    local command_decoder_hidden_outputs = {[0] = last_encoder_output}
    local command_decoder_out_outputs = {}

    local sampled = ''
    local sampled_tokens = {}

    for t = 1, opt.max_length do
        local command_decoder_output = models.command_decoder:forward(
            {command_decoder_inputs[t], command_decoder_hidden_outputs[t-1], encoder_outputs_padded})
        command_decoder_out_outputs[t] = command_decoder_output[1]
        command_decoder_hidden_outputs[t] = command_decoder_output[2][1]

        -- Choose most likely output
        out_max, out_max_index = command_decoder_out_outputs[t]:max(1)
        if out_max_index[1] == command_EOS then
            break
        end
        local output_argument_name = argument_index_to_value[out_max_index[1]]
        table.insert(sampled_tokens, output_argument_name)
        sampled = sampled .. output_argument_name
        if t < opt.max_length then sampled = sampled .. ' ' end

        -- Next decoder input is current output
        command_decoder_inputs[t + 1] = out_max_index
    end

    local last_command_decoder_output = command_decoder_hidden_outputs[#command_decoder_inputs - 1]

    print('sampled', sampled)
    return sampled

	-- -- For each argument pair

    -- local command = filter(sampled_tokens, function (t) return not isArgument(t) end)
    -- local sampled_arguments = filter(sampled_tokens, isArgument)
    -- local argument_decoder_inputs = map(sampled_arguments, function(argument_name)
    --     return torch.LongTensor({argument_name_to_index[argument_name]})
    -- end)

    -- local argument_decoder_hidden_outputs = {[0] = last_command_decoder_output}
    -- local attention_outputs = torch.zeros(#argument_decoder_inputs, opt.max_length)

    -- for t = 1, #argument_decoder_inputs do

    --     -- Forward through argument decoder

    --     argument_decoder_in_output = models.argument_decoder_in:forward(
    --         argument_decoder_inputs[t])
    --     argument_decoder_hidden_outputs[t], attention_output = unpack(models.argument_decoder_hidden:forward(
    --         {argument_decoder_in_output, argument_decoder_hidden_outputs[t-1], encoder_outputs_padded}))
    --     argument_decoder_out_output = models.argument_decoder_out:forward(
    --         argument_decoder_hidden_outputs[t])
    --     out_max, out_max_index = argument_decoder_out_output:max(1)

    --     attention_outputs[t] = attention_output

    --     local output_argument_value
    --     if out_max_index[1] <= n_argument_values then
    --         output_argument_value = argument_index_to_value[out_max_index[1]]
    --     else
    --         output_argument_value = 'UNK'
    --     end

    --     print(string.format("%s = %s", sampled_arguments[t], output_argument_value))
    --     table.insert(command, output_argument_value)
    --     -- command[sampled_arguments[t]:gsub('^%$', '')] = output_argument_value
    --     -- sampled = sampled .. ' ' .. output_argument_name
    --     -- command_decoder_inputs[t + 1] = out_max_index
    -- end

    -- print('argument decoder attention_outputs', attention_outputs)
    -- display.image(drawBig(attention_outputs, 30))

    -- return command
end

print(sample(sample_opt.sentence))

