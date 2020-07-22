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

parameters = svmLib.fit_polynomial_parameters()

print 'parameters fitted', parameters

print 'testing model...'

errors = svmLib.test_polynomial_model(parameters["c"], parameters["g"], parameters["d"])

print 'model tested', errors

mean = statistics.mean(errors)
standard_deviation = statistics.stdev(errors)

print 'mean', mean
print 'standard deviation', standard_deviation

f = open('svm_polynomial_kernel_results.txt', 'w')
f.write('C: ' + str(parameters['c']) + '\n')
f.write('gamma: ' + str(parameters['g']) + '\n')
f.write('degree: ' + str(parameters['d']) + '\n')
f.write('Mean: ' + str(mean) + '\n')
f.write('Standard deviation: ' + str(standard_deviation) + '\n')
f.close()
