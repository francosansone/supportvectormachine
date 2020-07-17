#!/usr/bin/python

import sys
import shutil
import os
import argparse


def write_data_in_files(folder_name, filename, data):
    ff = open(folder_name + '/' + filename, 'w')
    for line_fold in data:
        str_data = line_fold[0]
        for j in range(1, len(line_fold), 1):
            str_data += ',' + line_fold[j]
        str_data += '\n'
        ff.write(str_data)
    ff.close()
    file_res = open(folder_name + '/' + filename + '.svm', "w")
    ff = open(folder_name + '/' + filename)

    for line in ff:
        line = line.replace('\n', '')
        line = line.replace(' ', '')
        lines = line.split(',')
        str_line = lines.pop() + ' '
        for k in range(0, len(lines), 1):
            str_line += str(k + 1) + ':' + lines[k] + ' '
        file_res.write(str_line + '\n')
    ff.close()
    file_res.close()


parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str,
                    help="filename of dataset involved")
parser.add_argument("--input", type=int,
                    help="number of inputs")
args = parser.parse_args()

if not args.file:
    raise Exception("File name is needed")

if not args.input:
    raise Exception("Input number is needed")

arg = args.file
data_file = arg + '.data'
f = open(data_file)
names_file = arg + '.names'
n_input = args.input

lines_0 = []
lines_1 = []

print 'reading dataset...'

for line in f:
    _line = line.replace('\n', '').split(',')
    if int(_line[-1]) == 1:
        lines_1.append(_line)
    else:
        lines_0.append(_line)

f.close()

# random.shuffle(lines_0)
# random.shuffle(lines_1)

size_0 = len(lines_0) / 10
size_1 = len(lines_1) / 10
rest_0 = len(lines_0) % 10
rest_1 = len(lines_1) % 10

balance_0 = len(lines_0) / float((len(lines_0) + len(lines_1)))
balance_1 = len(lines_1) / float((len(lines_0) + len(lines_1)))

folds = []  # training_data, testing_data

nb_files = []

for i in range(0, 10, 1):
    add_0 = 0
    if rest_0 > 0:
        add_0 = 1
        rest_0 -= 1
    add_1 = 0
    if rest_1 > 0:
        add_1 = 1
        rest_1 -= 1
    from_0 = (i * size_0)
    to_0 = ((i + 1) * size_0 + add_0)
    from_1 = (i * size_1)
    to_1 = (i + 1) * size_1 + + add_1
    training_0_first = 0, from_0
    training_0_second = to_0 + 1, len(lines_0)
    training_1_first = 0, from_1
    training_1_second = to_1 + 1, len(lines_1)
    testing_data_0 = lines_0[from_0:to_0]
    testing_data_1 = lines_1[from_1:to_1]
    testing_data = testing_data_0 + testing_data_1
    training_data_0 = lines_0[training_0_first[0]:training_0_first[1]] + \
                      lines_0[training_0_second[0]:training_0_second[1]]
    training_data_1 = lines_1[training_1_first[0]:training_1_first[1]] + \
                      lines_1[training_1_second[0]:training_1_second[1]]
    training_data = training_data_0 + training_data_1
    training_data_size = len(training_data)
    testing_data_size = len(testing_data)
    nb_file_str = str(n_input) + '\n2\n' + \
                  str(training_data_size) + '\n' + \
                  str(int(training_data_size * 0.8)) + '\n' + \
                  str(testing_data_size) + '\n' + \
                  '0\n0\n'
    nb_files.append(nb_file_str)
    folds.append((training_data, testing_data))

print 'dataset read'

print 'writing new files...'

for i in range(0, 10, 1):
    folder_name = 'fold_' + str(i)
    basename = os.path.basename(arg)
    testing_file = basename + '.test'
    data_name_file = basename + '.data'
    names_file_name = os.path.basename(names_file)
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name, ignore_errors=True)
    os.mkdir(folder_name)
    shutil.copyfile(names_file, folder_name + '/' + names_file_name)

    write_data_in_files(folder_name, data_name_file, folds[i][0])

    write_data_in_files(folder_name, testing_file, folds[i][1])

    f = open(folder_name + '/' + basename + '.nb', 'w')
    f.write(nb_files[i])
    f.close()
print 'new files wrote'
