#!/usr/bin/python

import statistics
from lib import SvmLib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--debug", help="enable debug information",
                    action="store_true")
parser.add_argument("--file", type=str,
                    help="filename of dataset involved")
parser.add_argument("--iterations", type=int,
                    help="number of iterations to fit parameters")
parser.add_argument("--gamma", type=float,
                    help="gamma value")
args = parser.parse_args()

if not args.file:
    raise Exception("File name is needed")

iterations = 10
if args.iterations:
    iterations = args.iterations

gamma = 1.
if args.gamma:
    gamma = args.gamma

svmLib = SvmLib(args.file, debug=args.debug, iterations=iterations, gamma=gamma)

svmLib.clear_folds()

print 'fitting parameters...'

selected_c = svmLib.fit_linear_parameters()

print 'parameters fitted', selected_c

print 'testing model...'

errors = svmLib.test_model(selected_c)

print 'model tested', errors, selected_c

mean = statistics.mean(errors)
standard_deviation = statistics.stdev(errors)

print 'mean', mean
print 'standard deviation', standard_deviation

f = open('svm_linear_kernel_results.txt', 'w')
f.write('C: ' + str(selected_c) + '\n')
f.write('Mean: ' + str(mean) + '\n')
f.write('Standard deviation: ' + str(standard_deviation) + '\n')
f.close()
