require 'nn'
require 'nngraph'
require 'optim'
display = require 'display'

-- Parse command line arguments

cmd = torch.CmdLine()
cmd:text()
cmd:option('-hidden_size', 100, 'Hidden size of LSTM layer')
cmd:option('-glove_size', 100, 'Glove embedding size')
cmd:option('-dropout', 0.1, 'Dropout')
cmd:option('-learning_rate', 0.001, 'Learning rate')
cmd:option('-learning_rate_decay', 1e-5, 'Learning rate decay')
cmd:option('-max_length', 20, 'Maximum output length')
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
        local command_decoder_output = clones.command_decoder[t]:forward(
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
        sampled = sampled .. ' ' .. output_argument_name

        -- Next decoder input is current output
        command_decoder_inputs[t + 1] = out_max_index
    end

    local last_command_decoder_output = command_decoder_hidden_outputs[#command_decoder_inputs-1]

    print('sampled', sampled)
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
        local command_decoder_output = clones.command_decoder[t]:forward(
            {command_decoder_inputs[t], command_decoder_hidden_outputs[t-1], encoder_outputs_padded})
        command_decoder_out_outputs[t] = command_decoder_output[1]
        command_decoder_hidden_outputs[t] = command_decoder_output[2][1]
        loss = loss + clones.command_decoder_criterion[t]:forward(command_decoder_out_outputs[t], command_decoder_targets[t])
    end

    local last_command_decoder_output = command_decoder_hidden_outputs[#command_decoder_inputs]

	-- Backward through command decoder

    local d_command_decoder_out = {}
    local d_command_decoder_hidden = {[#command_decoder_inputs] = torch.zeros(opt.hidden_size)}
    local d_command_decoder_in = {}

    for t = #command_decoder_inputs, 1, -1 do
        -- decoder out < targets
        d_command_decoder_out[t] = clones.command_decoder_criterion[t]:backward(
            command_decoder_out_outputs[t], command_decoder_targets[t])

        -- -- < decoder
        d_command_decoder_in[t], d_command_decoder_hidden[t-1], _ = unpack(clones.command_decoder[t]:backward(
            {command_decoder_inputs[t], command_decoder_hidden_outputs[t-1], encoder_outputs_padded},
            {d_command_decoder_out[t], {d_command_decoder_hidden[t], torch.zeros(opt.max_length)}}
        ))
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

