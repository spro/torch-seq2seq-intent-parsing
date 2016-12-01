require 'nngraph'
require './gru'
require './model_utils'

if opt == nil then opt = torch.load('opt.t7') end

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

encoder_glove = nn.Linear(opt.glove_size, opt.hidden_size)()
encoder_glove_dropped = nn.Dropout(0.05)(encoder_glove)
encoder_gru_hidden_prev = nn.Identity()()
encoder_gru = GRU(opt.hidden_size, opt.hidden_size)({encoder_glove_dropped, encoder_gru_hidden_prev})
encoder_outputs = {encoder_gru}

encoder = nn.gModule({encoder_glove, encoder_gru_hidden_prev}, encoder_outputs)

-- Command decoder
-- =============================================================================
-- Decodes the last hidden state into a command (sequence of command, argument names)
--
-- Known:
-- * List of command and argument names ("setState", "$device", ...) and 
--
-- Inputs:
-- * Last output from itself (primed with EOS token)
-- * Last hidden state from itself
-- * Encoder hidden states
--
-- Outputs:
-- * Sequence of {command_name, argument_name, ...}

command_decoder_word = nn.LookupTable(n_command_words + 1, opt.hidden_size)()
command_decoder_hidden_prev = nn.Identity()()
all_encoder_outputs = nn.Identity()()
command_decoder_inputs = {
    command_decoder_word,
    command_decoder_hidden_prev,
    all_encoder_outputs
}

command_decoder_hidden = AttentionGRU(opt.hidden_size, opt.hidden_size, opt.max_length)(command_decoder_inputs)
command_decoder_selector = nn.Sequential()
    :add(nn.SelectTable(1))
    :add(nn.Linear(opt.hidden_size, n_command_words + 1))
    :add(nn.LogSoftMax())
command_decoder_selector = command_decoder_selector(command_decoder_hidden)
command_decoder_outputs = {command_decoder_selector, command_decoder_hidden}

command_decoder = nn.gModule(command_decoder_inputs, command_decoder_outputs)

-- Argument decoder
-- =============================================================================
-- Per placeholder output of the command decoder, finds the value(s) from the
-- input words to fill it
-- 
-- Inputs:
-- * Last output from itself (primed with EOS token)
-- * Last hidden state from itself
-- * Encoder hidden states

-- argument_decoder_command_word = nn.LookupTable(n_command_words + 1, opt.hidden_size)()
argument_decoder_command_word = nn.Identity()()
argument_decoder_input_word = nn.Linear(opt.glove_size, opt.hidden_size)()
argument_decoder_hidden_prev = nn.Identity()()
all_encoder_outputs = nn.Identity()()
argument_decoder_inputs = {
    argument_decoder_command_word,
    argument_decoder_input_word,
    argument_decoder_hidden_prev,
    all_encoder_outputs
}

joined_argument_words = nn.JoinTable(1)({
    nn.View(-1)(argument_decoder_command_word),
    argument_decoder_input_word
})
argument_decoder_hidden_inputs = {joined_argument_words, argument_decoder_hidden_prev, all_encoder_outputs} 
argument_decoder_hidden = AttentionGRU(opt.hidden_size * 2, opt.hidden_size, opt.max_length)(argument_decoder_hidden_inputs)

argument_decoder_selector = nn.Sequential()
    :add(nn.SelectTable(1))
    :add(nn.Linear(opt.hidden_size, 1))
    :add(nn.Sigmoid())
argument_decoder_selector = argument_decoder_selector(argument_decoder_hidden)
argument_decoder_outputs = {argument_decoder_selector, argument_decoder_hidden}

argument_decoder = nn.gModule(argument_decoder_inputs, argument_decoder_outputs)

-- Flattened parameters, clones per time step
-- =============================================================================

command_decoder_criterion = nn.ClassNLLCriterion()
argument_decoder_criterion = nn.BCECriterion()

models = {
    encoder=encoder,
    command_decoder=command_decoder,
    command_decoder_criterion=command_decoder_criterion,
    argument_decoder=argument_decoder,
    argument_decoder_criterion=argument_decoder_criterion,
}

params, grad_params = model_utils.combine_all_parameters(
    models.encoder,
    models.command_decoder,
    models.argument_decoder
)
params:uniform(-0.2, 0.2)

clones = mapObject(models, function(model)
    return model_utils.clone_many_times(model, opt.max_length)
end)

print('Model built')
