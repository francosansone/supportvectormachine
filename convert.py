#!/usr/bin/python

# <label> <index1>:<value1> <index2>:<value2> ...

import sys

file_res = open(str(sys.argv.pop()), "w")
f = open(str(sys.argv.pop()))

for line in f:
    line = line.replace('\n', '')
    line = line.replace(' ', '')
    lines = line.split(',')
    str_line = lines.pop() + ' '
    for i in range(1, len(lines), 1):
        str_line += str(i) + ':' + lines[i] + ' '
    file_res.write(str_line + '\n')
f.close()
file_res.close()
