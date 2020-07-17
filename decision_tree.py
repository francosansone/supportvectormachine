#!/usr/bin/python

import statistics
from lib import DTreeLib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--debug", help="enable debug information",
                    action="store_true")
parser.add_argument("--file", type=str,
                    help="filename of dataset involved")
args = parser.parse_args()

if not args.file:
    raise Exception("File name is needed")

dTreeLib = DTreeLib(args.file, debug=args.debug)

dTreeLib.clear_folds()

print 'testing model...'

errors = dTreeLib.train_and_test()

print 'model tested', errors

mean = statistics.mean(errors)
standard_deviation = statistics.pstdev(errors)

print 'mean', mean
print 'standard deviation', standard_deviation

f = open('decision_tree_results.txt', 'w')
f.write('Mean: ' + str(mean) + '\n')
f.write('Standard deviation: ' + str(standard_deviation) + '\n')
f.close()
