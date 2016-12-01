somata = require 'somata'
require 'sample'

service = somata.Service.create('sample', {
    sample=function(message, cb)
        if #message < 1 then
            cb("Input is too short")
        elseif #message > 75 then
            cb("Input is too long")
        else
            local command, args = sample(message)

            print('sampled', command, args)
            command = command:split(' ')
            method = command[1]
            args = {command[2], args}

            cb(nil, {results={}, parsed={method, args}, input=message})
        end
    end
}, {heartbeat=2000})

client = somata.Client.create(service.loop)
service:register()

