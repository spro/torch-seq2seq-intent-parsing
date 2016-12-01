require 'nn'
require 'nngraph'
require 'optim'
display = require 'display'
require './plot'

-- Parse command line arguments

cmd = torch.CmdLine()
cmd:text()
cmd:option('-hidden_size', 200, 'Hidden size of LSTM layer')
cmd:option('-glove_size', 100, 'Glove embedding size')
cmd:option('-dropout', 0.1, 'Dropout')
cmd:option('-learning_rate', 0.0002, 'Learning rate')
cmd:option('-learning_rate_decay', 1e-6, 'Learning rate decay')
cmd:option('-max_length', 20, 'Maximum output length')
cmd:option('-n_epochs', 10000, 'Number of epochs to train')
cmd:option('-series', 's1', 'Name of series')
opt = cmd:parse(arg)

require 'data'
require 'model'

-- Training
--------------------------------------------------------------------------------
-- Run a loop of optimization

n_epoch = 1

glove = torch.load('glove.t7')

function sample()
    -- Inputs and targets
    local sentence, encoder_inputs, command_decoder_inputs, _, argument_targets = unpack(makeSentence())
    local sentence_tokens = tokenize(sentence)
    print('--------\n[sample]', sentence)

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
        -- encoder_outputs_padded[t]:add(encoder_outputs_reverse[t])
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

    for arg_name, arg_value in pairs(command_argument_indexes) do
        arg_t = command_argument_ts[arg_name]
        sampled_tokens = ''
        arg_out_s = '{'

        -- Forward through argument decoder

        local argument_decoder_hidden_outputs = {[0] = last_encoder_output}
        -- local argument_decoder_hidden_outputs = {[0] = torch.zeros(opt.hidden_size)}
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

            -- Copy current token if > 0.5
            if argument_decoder_outputs[t][1] > 0.5 then
                arg_out_s = arg_out_s .. ' ' .. '1'
                sampled_tokens = sampled_tokens .. sentence_tokens[t] .. ' '
            else
                arg_out_s = arg_out_s .. ' ' .. '0'
            end
        end

        arg_out_s = arg_out_s .. ' }'
        sampled_full = sampled_full:gsub(arg_name, '( ' .. arg_name .. ' = ' .. sampled_tokens .. ')')
        print(arg_name, '~>', arg_out_s)
        local target = argument_targets[arg_name]
        if target ~= nil then
            print(arg_name, '=>', asString(target[{{1, #encoder_inputs}}]))
        else
            print('! no target')
        end
    end

    print('sampled', sampled_full)
end

function asString(t)
    s = ''
    for i = 1, t:size()[1] do
        s = s .. ' ' .. t[i]
    end
    return '{' .. s .. ' }'
end

sample()

function feval(params_)
    if params_ ~= params then
        params:copy(params_)
    end
    grad_params:zero()
    local loss = 0

    -- Inputs and targets

    local input_sentence, encoder_inputs, command_decoder_inputs, command_decoder_targets, argument_decoder_targets = unpack(makeSentence())

	-- Forward through encoder
    local encoder_outputs = {[0] = torch.zeros(opt.hidden_size)}
    for t = 1, #encoder_inputs do
        encoder_outputs[t] = clones.encoder[t]:forward({
            encoder_inputs[t],
            encoder_outputs[t-1]
        })
    end

    -- Pad encoder outputs with 0s

    last_encoder_output = encoder_outputs[#encoder_inputs]
    local encoder_outputs_padded = {}
    for t = 1, #encoder_outputs do
        table.insert(encoder_outputs_padded, encoder_outputs[t])
    end
    for t = 1, (opt.max_length - #encoder_outputs) do
        table.insert(encoder_outputs_padded, torch.zeros(opt.hidden_size))
    end

    -- Forward through command decoder

    local command_decoder_hidden_outputs = {[0] = last_encoder_output}
    local command_decoder_outputs = {}
    local command_decoder_output_indexes = {}
    local command_decoder_output_argument_indexes = {}

    for t = 1, #command_decoder_inputs do
        local command_decoder_output = clones.command_decoder[t]:forward({
            command_decoder_inputs[t],
            command_decoder_hidden_outputs[t-1],
            encoder_outputs_padded
        })
        command_decoder_outputs[t] = command_decoder_output[1]
        command_decoder_hidden_outputs[t] = command_decoder_output[2][1]
        _, command_decoder_output_indexes[t] = command_decoder_outputs[t]:max(1)
        loss = loss + clones.command_decoder_criterion[t]:forward(command_decoder_outputs[t], command_decoder_targets[t])
    end

    -- Arguments
    ---=========================================================================

    -- Set up gradients
    local d_encoder = torch.zeros(#encoder_inputs, opt.hidden_size)
    local d_command_decoder_hidden = {[#command_decoder_inputs] = torch.zeros(opt.hidden_size)}
    local d_command_decoder_in = {}
    local d_command_decoder_outputs = torch.zeros(opt.max_length, n_command_words + 1)

    -- First index command argument words to get related context

    local command_argument_ts = {}
    local command_argument_indexes = {}

    for t = 1, #command_decoder_targets - 1 do
        local command_index = command_decoder_targets[t][1]
        local command_word = command_index_to_word[command_index]
        if command_word:match(token_re) then
            command_argument_ts[command_word] = t
            command_argument_indexes[command_word] = torch.LongTensor({command_index})
        else
            -- We know they will have no gradient otherwise
            d_command_decoder_outputs[t] = torch.zeros(n_command_words + 1)
        end
    end

    -- For each command argument output ...

    for arg_name, arg_value in pairs(argument_decoder_targets) do
        local arg_t = command_argument_ts[arg_name]
        local arg_index = command_argument_indexes[arg_name]

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
            loss = loss + clones.argument_decoder_criterion[t]:forward(
                argument_decoder_outputs[t],
                arg_value[{{t}}]
            )
        end

        -- Backward through argument decoder

        local d_argument_decoder_out = {}
        local d_argument_decoder_all = {}
        local d_argument_decoder_hidden_command = {}
        local d_argument_decoder_hidden_encoder = {}
        local d_argument_decoder_hidden = {[#encoder_inputs] = torch.zeros(opt.hidden_size)}
        local d_argument_decoder_in = torch.zeros(opt.max_length, opt.hidden_size)

        for t = #encoder_inputs, 1, -1 do
            -- decoder out < targets
            d_argument_decoder_out[t] = clones.argument_decoder_criterion[t]:backward(
                argument_decoder_outputs[t],
                arg_value[{{t}}]
            )

            -- -- < decoder

            d_argument_decoder_all[t] = clones.argument_decoder[t]:backward(
                {
                    command_decoder_hidden_outputs[arg_t],
                    encoder_inputs[t],
                    argument_decoder_hidden_outputs[t-1],
                    encoder_outputs_padded
                },
                {
                    d_argument_decoder_out[t],
                    {
                        d_argument_decoder_hidden[t],
                        torch.zeros(opt.max_length)
                    }
                }
            )

            d_argument_decoder_in[t], d_encoder_out_t, d_argument_decoder_hidden[t-1], d_encoder_all = unpack(d_argument_decoder_all[t])

            -- Attention -> encoder gradients
            for tt = 1, #encoder_inputs do
                d_encoder[tt]:add(d_encoder_all[tt])
            end
        end

        -- Last encoder output was initial hidden state
        d_encoder[#encoder_inputs]:add(d_argument_decoder_hidden[0])
    end

	-- Backward through command decoder

    for t = #command_decoder_inputs, 1, -1 do
        -- decoder out < targets
        d_command_decoder_outputs_t = clones.command_decoder_criterion[t]:backward(
            command_decoder_outputs[t], command_decoder_targets[t])
        if d_command_decoder_outputs[t] == nil then
            d_command_decoder_outputs[t] = d_command_decoder_outputs_t
        else
            d_command_decoder_outputs[t]:add(d_command_decoder_outputs_t)
        end

        -- -- < decoder
        d_command_decoder = clones.command_decoder[t]:backward(
            {
                command_decoder_inputs[t],
                command_decoder_hidden_outputs[t-1],
                encoder_outputs_padded
            },
            {
                d_command_decoder_outputs[t],
                {
                    d_command_decoder_hidden[t],
                    torch.zeros(opt.max_length)
                }
            }
        )

        d_command_decoder_in[t], d_command_decoder_hidden[t-1], d_encoder_all = unpack(d_command_decoder)
        for tt = 1, #encoder_inputs do
            d_encoder[tt]:add(d_encoder_all[tt])
        end
    end

    -- Last encoder output was initial hidden state
    d_encoder[#encoder_inputs]:add(d_command_decoder_hidden[0])

    -- Backward through encoder

    for t = #encoder_inputs, 1, -1 do
        local _, d_encoder_t_1 = unpack(clones.encoder[t]:backward(
            {encoder_inputs[t], encoder_outputs[t-1] or torch.zeros(opt.hidden_size)},
            d_encoder[t]
        ))
        if t > 1 then
            d_encoder[t-1] = d_encoder_t_1
        end
    end

	return loss, grad_params
end

losses = {}
loss_sofar = 0
learning_rates = {}
plot_every = 100
sample_every = 50
save_every = 5000

optim_state = {
    learningRate = opt.learning_rate,
    learningRateDecay = opt.learning_rate_decay
}

function save()
    print('Saving...')
    torch.save('models.t7', models)
    torch.save('opt.t7', opt)
end

LOSS_PLOT_CUTOFF = 5 * plot_every

print(string.format("Training for %s epochs...", opt.n_epochs))
for n_epoch = 1, opt.n_epochs do
    local _, loss = optim.adam(feval, params, optim_state)
    loss_sofar = loss_sofar + loss[1]

    -- Plot every plot_every
    if n_epoch % plot_every == 0 then
        if loss_sofar > 0 and loss_sofar < LOSS_PLOT_CUTOFF then
            plot({x=n_epoch, y=loss_sofar/plot_every, series=opt.series})
        end
        loss_sofar = 0
    end

    -- Sample every sample_every
    if n_epoch % sample_every == 0 then
        sample()
        print(n_epoch, loss[1])
    end

    -- Save every save_every
    if n_epoch % save_every == 0 then
        save()
    end
end


