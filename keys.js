const express = require('express')
const app = express()
const fs = require("fs");

var keys = []
var stream = fs.createReadStream("data-students.json", { flags: 'r', encoding: 'utf-8' });

app.get('/', function(req, res) {
    var buf = '';

    stream.on('data', function(d) {
        buf += d.toString(); // when data is read, stash it in a string buffer
        pump(); // then process the buffer

    })
    stream.on("close", () => {
        console.log(keys)
    })

    console.log(keys)

    function pump() {
        var pos;

        while ((pos = buf.indexOf('\n')) >= 0) { // keep going while there's a newline somewhere in the buffer
            if (pos == 0) { // if there's more than one newline in a row, the buffer will now start with a newline
                buf = buf.slice(1); // discard it
                continue; // so that the next iteration will start with data
            }
            processLine(buf.slice(0, pos)); // hand off the line
            buf = buf.slice(pos + 1); // and slice the processed data off the buffer
        }
        stream.close()
    }

    function processLine(line) { // here's where we do something with a line

        if (line[line.length - 1] == '\r') line = line.substr(0, line.length - 1); // discard CR (0x0D)

        if (line.length > 0) { // ignore empty lines
            var obj = JSON.parse(line); // parse the JSON
            Object.keys(obj).forEach(element => {
                if (!keys.includes(element)) {
                    keys.push(element)
                }
            });
        }
    }
})

app.listen(3000, function() {
    console.log('Example app listening on port 3000!')
})