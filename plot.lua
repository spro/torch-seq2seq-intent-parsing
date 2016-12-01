http = require 'socket.http'
json = require 'cjson'
ltn12 = require 'ltn12'

function plot(point, series)
    local point = json.encode(point)
    local content_length = #point
    http.request({
        url = 'http://live-graph.dev/points.json',
        method = 'POST',
        headers = {
            ['content-length'] = content_length,
            ['content-type'] = 'application/json'
        },
        source = ltn12.source.string(point)
    })
end

