require 'nn'
require 'nngraph'
require 'optim'
display = require 'display'

-- Parse command line arguments

cmd = torch.CmdLine()
cmd:text()
cmd:option('-hidden_size', 100, 'Hidden size of LSTM layer')
cmd:option('-dropout', 0.1, 'Dropout')
cmd:option('-learning_rate', 0.001, 'Learning rate')
cmd:option('-learning_rate_decay', 1e-4, 'Learning rate decay')
cmd:option('-max_length', 15, 'Maximum output length')
cmd:option('-n_epochs', 100000, 'Number of epochs to train')
cmd:option('-win', 'losses', 'Name of display window')

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
    local sentence, encoder_inputs, command_decoder_inputs, command_decoder_targets, argument_decoder_inputs, argument_decoder_targets = makeSentence()
    print('--------\n[sample]', sentence)
    -- print('.. command decoder inputs'); pt(command_decoder_inputs)
    -- print('.. command decoder targets', command_decoder_targets)
    -- print('** argument decoder inputs'); pt(argument_decoder_inputs)
    -- print('** argument decoder targets', argument_decoder_targets)

	-- Forward pass
    -- -------------------------------------------------------------------------

    -- Through encoder
    local encoder_outputs = {[0] = torch.zeros(opt.hidden_size):double()}
    for t = 1, #encoder_inputs do
        encoder_outputs[t] = clones.encoder[t]:forward({encoder_inputs[t][1], encoder_inputs[t][2], encoder_outputs[t-1]})
    end
    last_encoder_output = encoder_outputs[#encoder_inputs]

    -- Pad encoder outputs with 0s

    local encoder_outputs_padded = {}
    for t = 1, #encoder_outputs do
        table.insert(encoder_outputs_padded, encoder_outputs[t])
    end
    for t = 1, (opt.max_length - #encoder_outputs) do
        table.insert(encoder_outputs_padded, torch.zeros(opt.hidden_size))
    end

    -- Through command decoder
    local command_decoder_inputs = {[1] = torch.LongTensor({command_EOS})}
    local command_decoder_in_outputs = {}
    local command_decoder_hidden_outputs = {[0] = last_encoder_output}
    local command_decoder_out_outputs = {}

    local sampled = ''
    local sampled_tokens = {}

    for t = 1, opt.max_length do
        command_decoder_in_outputs[t] = clones.command_decoder_in[t]:forward(
            command_decoder_inputs[t])
        local command_decoder_hidden_output = clones.command_decoder_hidden[t]:forward(
            {command_decoder_in_outputs[t], command_decoder_hidden_outputs[t-1], encoder_outputs_padded})
        command_decoder_hidden_outputs[t] = command_decoder_hidden_output[1]
        local command_decoder_hidden_attention = command_decoder_hidden_output[2]
        command_decoder_out_outputs[t] = clones.command_decoder_out[t]:forward(
            command_decoder_hidden_outputs[t])

        out_max, out_max_index = command_decoder_out_outputs[t]:max(1)
        if out_max_index[1] == command_EOS then
            break
        end
        local output_argument_name = argument_index_to_value[out_max_index[1]]
        table.insert(sampled_tokens, output_argument_name)
        sampled = sampled .. ' ' .. output_argument_name
        command_decoder_inputs[t + 1] = out_max_index
    end

    local last_command_decoder_output = command_decoder_hidden_outputs[#command_decoder_inputs-1]

    print('sampled', sampled)

    -- return nil

	-- -- For each argument pair

    -- local sampled_arguments = filter(sampled_tokens, isArgument)
    -- local argument_decoder_inputs = map(sampled_arguments, function(argument_name)
    --     return torch.LongTensor({argument_name_to_index[argument_name]})
    -- end)

    -- local argument_decoder_hidden_outputs = {[0] = last_command_decoder_output}

    -- for t = 1, #argument_decoder_inputs do

    --     -- Forward through argument decoder

    --     argument_decoder_in_output = clones.argument_decoder_in[t]:forward(
    --         argument_decoder_inputs[t])
    --     argument_decoder_hidden_outputs[t], _ = unpack(clones.argument_decoder_hidden[t]:forward(
    --         {argument_decoder_in_output, torch.zeros(opt.hidden_size), encoder_outputs_padded}))
    --     argument_decoder_out_output = clones.argument_decoder_out[t]:forward(
    --         argument_decoder_hidden_outputs[t])
    --     out_max, out_max_index = argument_decoder_out_output:max(1)
    --     local output_argument_value
    --     if out_max_index[1] <= n_argument_values then
    --         output_argument_value = argument_index_to_value[out_max_index[1]]
    --     else
    --         output_argument_value = 'UNK'
    --     end
    --     print(string.format("%s = %s", sampled_arguments[t], output_argument_value))
    --     -- sampled = sampled .. ' ' .. output_argument_name
    --     -- command_decoder_inputs[t + 1] = out_max_index
    -- end

end

sample()

function feval(params_)
    if params_ ~= params then
        params:copy(params_)
    end
    grad_params:zero()
    local loss = 0

    -- Inputs and targets

    local input_sentence, encoder_inputs, command_decoder_inputs, command_decoder_targets, argument_decoder_inputs, argument_decoder_targets = makeSentence()

	-- Forward through encoder

    local encoder_outputs = {[0] = torch.zeros(opt.hidden_size)}
    for t = 1, #encoder_inputs do
        encoder_outputs[t] = clones.encoder[t]:forward({encoder_inputs[t][1], encoder_inputs[t][2], encoder_outputs[t-1]})
    end
    last_encoder_output = encoder_outputs[#encoder_inputs]

    -- Pad encoder outputs with 0s

    local encoder_outputs_padded = {}
    for t = 1, #encoder_outputs do
        table.insert(encoder_outputs_padded, encoder_outputs[t])
    end
    for t = 1, (opt.max_length - #encoder_outputs) do
        table.insert(encoder_outputs_padded, torch.zeros(opt.hidden_size))
    end

    -- Forward through command decoder

    local command_decoder_in_outputs = {}
    local command_decoder_hidden_outputs = {[0] = last_encoder_output}
    local command_decoder_out_outputs = {}

    for t = 1, #command_decoder_inputs do
        command_decoder_in_outputs[t] = clones.command_decoder_in[t]:forward(
            command_decoder_inputs[t])
        command_decoder_hidden_outputs[t], _ = unpack(clones.command_decoder_hidden[t]:forward(
            {command_decoder_in_outputs[t], command_decoder_hidden_outputs[t-1], encoder_outputs_padded}))
        command_decoder_out_outputs[t] = clones.command_decoder_out[t]:forward(
            command_decoder_hidden_outputs[t])
        loss = loss + clones.command_decoder_criterion[t]:forward(command_decoder_out_outputs[t], command_decoder_targets[t])
    end

    local last_command_decoder_output = command_decoder_hidden_outputs[#command_decoder_inputs]

	-- -- For each argument pair

    -- local argument_decoder_in_outputs = {}
    -- local argument_decoder_hidden_outputs = {[0] = last_command_decoder_output}
    -- local argument_decoder_out_outputs = {}

    -- -- Forward through argument decoder
    -- for t = 1, #argument_decoder_inputs do

        -- argument_decoder_in_outputs[t] = clones.argument_decoder_in[t]:forward(
            -- argument_decoder_inputs[t])
        -- argument_decoder_hidden_outputs[t], _ = unpack(clones.argument_decoder_hidden[t]:forward(
            -- {argument_decoder_in_outputs[t], torch.zeros(opt.hidden_size), encoder_outputs_padded}))
        -- argument_decoder_out_outputs[t] = clones.argument_decoder_out[t]:forward(
            -- argument_decoder_hidden_outputs[t])

        -- loss = loss + clones.argument_decoder_criterion[t]:forward(
            -- argument_decoder_out_outputs[t], argument_decoder_targets[t])
    -- end

    -- local d_argument_decoder_hidden = {[#argument_decoder_inputs] = torch.zeros(opt.hidden_size)}

    -- -- Backward through argument decoder
    -- for t = #argument_decoder_inputs, 1, -1 do

        -- -- decoder out < targets
        -- d_argument_decoder_out = clones.argument_decoder_criterion[t]:backward(
            -- argument_decoder_out_outputs[t], argument_decoder_targets[t])
        -- -- decoder hidden < decoder out
        -- d_argument_decoder_hidden[t]:add(clones.argument_decoder_out[t]:backward(
            -- argument_decoder_hidden_outputs[t], d_argument_decoder_out))
        -- -- decoder in < decoder hidden
        -- d_argument_decoder_in, d_argument_decoder_hidden[t-1], _ = unpack(clones.argument_decoder_hidden[t]:backward(
            -- {argument_decoder_in_outputs[t], torch.zeros(opt.hidden_size), encoder_outputs_padded},
            -- {d_argument_decoder_hidden[t], torch.zeros(opt.max_length)}
        -- ))
        -- -- < decoder in
        -- clones.argument_decoder_in[t]:backward(
            -- argument_decoder_inputs[t], d_argument_decoder_in)
    -- end

    -- -- print('d argument decoder hidden', d_argument_decoder_hidden)

    -- local sum_d_argument_decoder_hidden = torch.zeros(opt.hidden_size)
    -- for t = 0, #argument_decoder_inputs do
        -- sum_d_argument_decoder_hidden:add(d_argument_decoder_hidden[t])
    -- end

	-- Backward through command decoder

    local d_command_decoder_out = {}
    -- local d_command_decoder_hidden = {[#command_decoder_inputs] = sum_d_argument_decoder_hidden}
    local d_command_decoder_hidden = {[#command_decoder_inputs] = torch.zeros(opt.hidden_size)}
    local d_command_decoder_in = {}

    for t = #command_decoder_inputs, 1, -1 do
        -- decoder out < targets
        d_command_decoder_out[t] = clones.command_decoder_criterion[t]:backward(
            command_decoder_out_outputs[t], command_decoder_targets[t])
        -- decoder hidden < decoder out
        d_command_decoder_hidden[t]:add(clones.command_decoder_out[t]:backward(
            command_decoder_hidden_outputs[t], d_command_decoder_out[t]))
        -- decoder in < decoder hidden
        d_command_decoder_in[t], d_command_decoder_hidden[t-1], _ = unpack(clones.command_decoder_hidden[t]:backward(
            {command_decoder_in_outputs[t], command_decoder_hidden_outputs[t-1], encoder_outputs_padded},
            {d_command_decoder_hidden[t], torch.zeros(opt.max_length)}
        ))
        -- < decoder in
        clones.command_decoder_in[t]:backward(
            command_decoder_inputs[t], d_command_decoder_in[t])
    end

    -- Backward through encoder

    local d_encoder = {[#encoder_inputs] = d_command_decoder_hidden[0]}
    for t = #encoder_inputs, 1, -1 do
        _, _, d_encoder[t-1] = unpack(clones.encoder[t]:backward(
            {encoder_inputs[t][1], encoder_inputs[t][2], encoder_outputs[t-1] or torch.zeros(opt.hidden_size)},
            d_encoder[t]
        ))
    end

	return loss, grad_params
end

losses = {}
loss_sofar = 0
learning_rates = {}
plot_every = 10
sample_every = 500
save_every = 10000

optim_state = {
    learningRate = opt.learning_rate,
    learningRateDecay = opt.learning_rate_decay
}

function save()
    print('Saving...')
    torch.save('models.t7', models)
    torch.save('opt.t7', opt)
end

print(string.format("Training for %s epochs...", opt.n_epochs))
for n_epoch = 1, opt.n_epochs do
    local _, loss = optim.adam(feval, params, optim_state)
    loss_sofar = loss_sofar + loss[1]

    -- Plot every plot_every
    if n_epoch % plot_every == 0 then
        if loss_sofar > 0 and loss_sofar < 9999 then
            table.insert(losses, {n_epoch, loss_sofar / plot_every})
            display.plot(losses, {win=opt.win})
        else
            print('loss', loss)
        end
        loss_sofar = 0
    end

    -- Sample every sample_every
    if n_epoch % sample_every == 0 then
        sample()
    end

    -- Save every save_every
    if n_epoch % save_every == 0 then
        save()
    end
end

