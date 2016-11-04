require 'nn'
require 'nngraph'
model_utils = require './model_utils'

function GRU(input_size, hidden_size)
    local input = nn.View(-1)()
    local prev_h = nn.View(-1)()
    local inputs = {input, prev_h}

	function makeGate(i, h)
		local i2h = nn.Linear(input_size, hidden_size)(i)
		local h2h = nn.Linear(hidden_size, hidden_size)(h)
		return nn.CAddTable()({i2h, h2h})
	end

    local z = nn.Sigmoid()(makeGate(input, prev_h))
    local r = nn.Sigmoid()(makeGate(input, prev_h))

    local reset_prev_h = nn.CMulTable()({r, prev_h})
    local h_candidate = nn.Tanh()(makeGate(input, reset_prev_h))

    local nz = nn.AddConstant(1)(nn.MulConstant(-1)(z))
    local zh = nn.CMulTable()({z, h_candidate})
    local nzh = nn.CMulTable()({nz, prev_h})
    local next_h = nn.CAddTable()({zh, nzh})

    local outputs = {next_h}
    return nn.gModule(inputs, outputs)
end

function AttentionGRU(input_size, hidden_size, max_length)
    -- Inputs
    local input = nn.View(-1)()
    local prev_h = nn.View(-1)()
    local encs = nn.Identity()()
    local inputs = {input, prev_h, encs}

    -- Calculate attention
    local encs_sum = nn.View(-1)(nn.CAddTable()(encs))
    local full_context = nn.JoinTable(1)({input, prev_h, encs_sum})
    local attn_context = nn.Linear(input_size + hidden_size * 2, max_length)(full_context)
    local attn_coef = nn.SoftMax()(attn_context)
    local attn = nn.MixtureTable()({attn_coef, encs})

    function makeGate(i, h)
        local i2h = nn.Linear(input_size, hidden_size)(i)
        local h2h = nn.Linear(hidden_size, hidden_size)(h)
        local a2h = nn.Linear(hidden_size, hidden_size)(attn) -- Adds attention output to all gates
        return nn.CAddTable()({i2h, h2h, a2h})
    end

    -- Regular GRU computation
    local z = nn.Sigmoid()(makeGate(input, prev_h))
    local r = nn.Sigmoid()(makeGate(input, prev_h))

    local reset_prev_h = nn.CMulTable()({r, prev_h})
    local h_candidate = nn.Tanh()(makeGate(input, reset_prev_h))

    local nz = nn.AddConstant(1)(nn.MulConstant(-1)(z))
    local zh = nn.CMulTable()({z, h_candidate})
    local nzh = nn.CMulTable()({nz, prev_h})
    local next_h = nn.CAddTable()({zh, nzh})

    -- Output attention alongside state for later analysis
    local outputs = {next_h, attn_coef}
    return nn.gModule(inputs, outputs)
end

-- Encoder
-- =============================================================================
-- Encodes the input sentence (sequence of words) as L x H dimensional hidden state
--
-- Known:
-- * GloVe vectors
--
-- Inputs:
-- * Sentence as sequence of words as GloVe vectors
--
-- Outputs:
-- * Sequence of hidden states

encoder_glove = nn.Linear(opt.hidden_size, opt.hidden_size)()
encoder_lookup = nn.LookupTable(n_input_words + 1, opt.hidden_size)()
encoder_lookup_flat = nn.View(-1)(encoder_lookup)
-- Combine glove and lookup
encoder_word = nn.JoinTable(1)({encoder_glove, encoder_lookup_flat})
encoder_gru_hidden_prev = nn.Identity()()
encoder_inputs = {encoder_glove, encoder_lookup, encoder_gru_hidden_prev}
encoder_gru_inputs = {encoder_word, encoder_gru_hidden_prev}
encoder_gru = GRU(opt.hidden_size * 2, opt.hidden_size)
encoder_outputs = {encoder_gru(encoder_gru_inputs)}

encoder = nn.gModule(encoder_inputs, encoder_outputs)

-- Command decoder
-- =============================================================================
-- Decodes the last hidden state into a command (sequence of command, argument names)
--
-- Known:
-- * List of command names ("setState", ...) and 
-- * List of argument names ("device_name", ...)
--
-- Inputs:
-- * Encoder hidden state(s)
--
-- Outputs:
-- * Sequence of {command_name, argument_name, ...}

command_decoder_in = nn.LookupTable(n_argument_values + 1, opt.hidden_size)
command_decoder_hidden = AttentionGRU(opt.hidden_size, opt.hidden_size, opt.max_length)
command_decoder_out = nn.Sequential()
    :add(nn.Linear(opt.hidden_size, n_argument_values + 1))
    :add(nn.LogSoftMax())

-- Argument decoder
-- =============================================================================
-- Per argument, outputs a value from set of known possible argument values (e.g. device names)
--
-- Known:
-- * List of argument values ("office light", ...)
--
-- Inputs:
-- * Encoder hidden states
-- * Decoded command (first output of command decoder)
-- * Decoded argument (per 2nd + output of command decoder)
--
-- Outputs:
-- * Single argument value from list of known argument values

-- argument_decoder_in = AttentionGRU(opt.hidden_size, opt.hidden_size, opt.max_length)
argument_decoder_in = nn.Sequential()
    :add(nn.ParallelTable()
        :add(nn.LookupTable(n_argument_values, opt.hidden_size))
        :add(nn.View(1, -1))
    )
    :add(nn.JoinTable(2))
argument_decoder_hidden = AttentionGRU(opt.hidden_size * 2, opt.hidden_size, opt.max_length)
argument_decoder_out = nn.Sequential()
    :add(nn.Linear(opt.hidden_size, n_argument_values))
    :add(nn.LogSoftMax())

-- TODO
-- Separate freeform argument decoder
-- =============================================================================
-- Per argument, decodes whether each word corresponds to this value
-- Or could the other argument decoder output a COPY "command"?
--
-- Inputs:
-- * Encoder hidden states
-- * Decoded command
-- * Decoded argument (per argument)
-- * Input word (per step)
--
-- Outputs:
-- * Sequence of {true, false} over input words, to filter by and concatenate into a string

command_decoder_criterion = nn.ClassNLLCriterion()
argument_decoder_criterion = nn.ClassNLLCriterion()

models = {
    encoder=encoder,
    -- encoder_in=encoder_in,
    -- encoder_hidden=encoder_hidden,

    command_decoder_in=command_decoder_in,
    command_decoder_hidden=command_decoder_hidden,
    command_decoder_out=command_decoder_out,

    argument_decoder_in=argument_decoder_in,
    argument_decoder_hidden=argument_decoder_hidden,
    argument_decoder_out=argument_decoder_out,

    command_decoder_criterion=command_decoder_criterion,
    argument_decoder_criterion=argument_decoder_criterion,
}

params, grad_params = model_utils.combine_all_parameters(
    models.encoder,
    -- models.encoder_in,
    -- models.encoder_hidden,
    models.command_decoder_in,
    models.command_decoder_hidden,
    models.command_decoder_out,
    models.argument_decoder_in,
    models.argument_decoder_hidden,
    models.argument_decoder_out
)
params:uniform(-0.1, 0.1)

clones = mapObject(models, function(model)
    return model_utils.clone_many_times(model, opt.max_length)
end)

print('Model built')
