
sentence_command_templates = {
    {"turn $light.state the $light.device",   {"lights", "setState", "$light.device", "$light.state"}},
    {"turn the $light.device $light.state",   {"lights", "setState", "$light.device", "$light.state"}},
    {"make the $light.device $light.state",   {"lights", "setState", "$light.device", "$light.state"}},
    {"toggle the $light.device",              {"lights", "toggleState", "$light.device"}},
    {"toggle $light.group",                   {"lights", "toggleStates", "$light.group"}},
    {"turn $light.state $light.group",        {"lights", "setStates", "$light.group", "$light.state"}},
    {"turn $light.group $light.state",        {"lights", "setStates", "$light.group", "$light.state"}},
    {"make $light.group $light.state",        {"lights", "setStates", "$light.group", "$light.state"}},
    {"is the $light.device on?",              {"lights", "getState", "$light.device"}},
    {"how is the $light.device?",             {"lights", "getState", "$light.device"}},

    {"turn $switch.state the $switch.device", {"switches", "setState", "$switch.device", "$switch.state"}},
    {"turn the $switch.device $switch.state", {"switches", "setState", "$switch.device", "$switch.state"}},
    {"is the $switch.device on?",             {"switches", "getState", "$switch.device"}},
    {"how is the $switch.device?",            {"switches", "getState", "$switch.device"}},
    {"make tea",                              {"switches", "setState", "tea", "on"}},

    {"is the $contact.device $contact.state?",{"sensors", "getState", "$contact.device"}},

    {"stop $juicebox.music",                  {"juicebox", "stop"}},
    {"turn off $juicebox.music",              {"juicebox", "stop"}},
    {"turn $juicebox.music $juicebox.volume", {"juicebox", "setVolume", "$juicebox.volume"}},
    {"turn it $juicebox.volume",              {"juicebox", "setVolume", "$juicebox.volume"}},
    {"make $juicebox.music $juicebox.volume", {"juicebox", "setVolume", "$juicebox.volume"}},
    {"play something",                        {"juicebox", "play"}},
    {"play some music",                       {"juicebox", "play"}},
    {"start juicebox",                        {"juicebox", "play"}},
    {"play the next song on juicebox",        {"juicebox", "next"}},
    {"skip this song",                        {"juicebox", "next"}},
    {"skip this",                             {"juicebox", "next"}},

    {"what time is it?",                      {"time", "getTime"}},
    {"how much is $coin.symbol?",             {"price", "getPrice", "$coin.symbol"}},
    {"what's the price of $coin.symbol?",     {"price", "getPrice", "$coin.symbol"}},

    {"what's the $weather.state in $weather.location", {"weather", "getState", "$weather.location", "$weather.state"}},
    {"what's the weather in $weather.location",        {"weather", "getState", "$weather.location"}},
    {"how $weather.state is it in $weather.location",  {"weather", "getState", "$weather.location", "$weather.state"}},
    {"how is it in $weather.location",                 {"weather", "getState", "$weather.location"}},

}

argument_values = {
    light = {
        device = {
            office_light = {"office light", "light in the office", "office hue"},
            living_room_light = {"living room light", "light in the living room"},
        },
        group = {
            all_lights = {"all the lights", "all of the lights", "the lights", "every light"},
            seans_room_lights = {"my lights", "the lights in my room", "all my lights"},
        },
        state = {
            on = {"on"},
            off = {"off"},
            up = {"up", "brighter", "more", "brighter"},
            down = {"down", "lower", "less", "less bright"},
            red = {"red"},
            blue = {"blue"},
            green = {"green"},
            purple = {"purple"},
            orange = {"orange"},
            yellow = {"yellow"},
            white = {"white", "normal"},
        },
    },

    switch = {
        device = {
            tea = {"tea", "hotplate", "tea switch", "teapot"},
        },
        state = {
            on = {"on"},
            off = {"off"},
        },
    },

    contact = {
        device = {
            door = {"door", "front door", "door switch", "door sensor"},
        },
        state = {
            open = {"open"},
            closed = {"closed"},
        },
    },

    weather = {
        location = {
            office = {"office", "the office"},
            living_room = {"living room", "the living room", "the den"},
            san_francisco = {"san francisco", "sf", "the city", "downtown"},
            mountain_view = {"mountain view", "the bay", "mv", "google"},
        },

        state = {
            temperature = {"temperature", "warm", "cold"},
            humidity = {"humidity", "humid"},
            conditions = {"conditions", "the sky"},
            pressure = {"pressure", "air pressure"},
        },
    },

    juicebox = {
        music = {
            placeholder = {"music", "the music", "the song", "juicebox", "jukebox", "this shit"},
        },

        volume = {
            up = {"up", "louder", "higher"},
            down = {"down", "quieter"},
            low = {"low", "quiet"},
            middle = {"middle", "fifty percent", "half"},
            high = {"high", "loud"},
        },
    },

    coin = {
        symbol = {
            btc = {"btc", "bitcoin"},
            eth = {"eth", "ethereum"},
        }
    },

}

noise_pre = {
    "hey there", "hey maia", "maia", "please", "maia, can you please", "maia would you", "yo", "so"
}

noise_post = {
    "pretty please", "please", "maia", "ok?", "if you would", ", or else", "now", "if that's ok"
}

