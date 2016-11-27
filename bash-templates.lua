
sentence_command_templates = {
    {"$commands.list in $files.directory",   {"ls", "$files.directory"}},
    {"$commands.list in $files.directory by $files.sort",   {"ls", "$files.sort", "$files.directory"}},
    {"find the $files.sort files in $files.directory",   {"ls", "$files.sort", "$files.directory"}},
    {"find all $files.ext files in $files.directory",   {"find", "$files.directory", "-name", "$files.ext"}},
    {"$commands.remove $files.directory",   {"rm", "-r", "$files.directory"}},
    {"$commands.remove $files.file",   {"rm", "$files.file"}},
    {"$commands.move $files.file to $files.directory",   {"mv", "$files.file", "$files.directory"}},
    {"$commands.copy $files.file to $files.directory",   {"cp", "$files.file", "$files.directory"}},
    {"$commands.open $apps.app",   {"open", "$apps.app"}},
    {"$commands.read $files.file",   {"cat", "$files.file"}},
    {"$commands.size $files.directory",   {"du", "$files.directory"}},
    {"$commands.date", {"date"}},
    {"count $words.count of $files.file", {"wc", "$files.file"}},
}

argument_values = {
    words = {
        count = {_ = {"lines", "words", "bytes"}},
    },

    commands = {
        date = {_ = {"date", "time", "what time is it", "what day is it?", "what is it today?"}},
        list = {_ = {"list", "find files", "show everything"}},
        read = {_ = {"read", "print out"}},
        move = {_ = {"move"}},
        copy = {_ = {"copy"}},
        remove = {_ = {"remove", "delete"}},
        open = {_ = {"open", "start"}},
        size = {_ = {"how big is", "how much size is"}},
    },

    files = {
        archivename = {_ = {"files.tgz", "ball.tar.gz", "stuff.tgz"}},

        file = {
            ["~/.ssh/config"] = {"my ssh config"},
            ["~/Notes.txt"] = {"my notes file", "notes"},
            ["~/.aliases"] = {"my aliases file"},
            ["~/.aliases"] = {"my aliases file"},
            ["~/.bash_history"] = {"bash history", "my history file"},
            ["/var/nginx/conf/nginx.conf"] = {"nginx config", "the server configuration"},
            ["/etc/redis/redis.conf"] = {"redis config", "the cache configuration"},
            ["/usr/local/var/postgres/postgresql.conf"] = {"psql config", "postgres config", "the database configuration"},
        },

        directory = {
            ["~"] = {"home", "my home directory", "my files", "my main folder"},
            ["~/Projects"] = {"my projects dir", "projects", "the projects folder"},
            ["~/Desktop"] = {"my desktop", "the desktop"},
            ["~/Downloads"] = {"my downloads folder", "downloads"},
            ["/"] = {"root", "the root directory", "the base dir"},
            ["/Applications"] = {"applications", "apps", "the main applications folder"},
            ["/tmp"] = {"temporary", "temp"},
        },

        sort = {
            ["-lt"] = {"oldest"},
            ["-ltr"] = {"newest", "most recent"},
            ["-lS"] = {"smallest", "small"},
            ["-lSr"] = {"biggest", "large"},
        },

        ext = {
            ["*.txt"] = {"text", "plain text"},
            ["*.conf"] = {"config", "configuration"},
            ["*.swp"] = {"vim temp", "temporary editor"},
            ["*.py"] = {"python"},
            ["*.js"] = {"js", "javscript"},
            ["*.coffee"] = {"coffee", "coffee script", "coffeescript"},
        },
    },

    apps = {
        app = {
            ["/Applications/iTunes.app"] = {"itunes", "music"},
            ["/Applications/Sketch.app"] = {"sketch", "the vector editor"},
            ["/Applications/Terminal.app"] = {"terminal", "the command line"},
        },
    },
}

noise_pre = {
    "hello computer", "hi, ", "could you", "please", "computer, will you please", "i would like you to", "yo"
}

noise_post = {
    "pretty please", "please", "ok?", "if you would", ", or else", "now", "if that's ok"
}

