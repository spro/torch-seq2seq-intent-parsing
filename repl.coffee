qrepl = require 'qrepl'
somata = require 'somata'
client = new somata.Client
qrepl 'sample', (line, cb) ->
    client.remote 'sample', 'sample', line.trim(), cb

