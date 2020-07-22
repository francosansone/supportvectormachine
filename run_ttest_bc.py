#!/usr/bin/python

import math
import statistics

naive_bayes = [0.05172414, 0.12068965000000001, 0.0877193, 0.10526316, 0.07017544, 0.01754386, 0.07017544, 0.01785714,
               0.10714285999999999, 0.03571429]

svm_linear = [0.0, 0.1724, 0.0351, 0.0877, 0.0351, 0.0175, 0.0702, 0.1071, 0.0179, 0.0179]

svm_pol = [0.0172, 0.1379, 0.0526, 0.0526, 0.0351, 0.0351, 0.0526, 0.0179, 0.1964, 0.0714]

diffs = [naive_bayes[i] - svm_linear[i] for i in range(0, 10)]

d = statistics.mean(diffs)

s_d = statistics.stdev(diffs) / math.sqrt(10)

t = d / s_d

print "svm linear tree and naive bayes", t

_diffs = [svm_pol[i] - svm_linear[i] for i in range(0, 10)]

_d = statistics.mean(_diffs)

_s_d = statistics.stdev(_diffs) / math.sqrt(10)

_t = _d / _s_d

print "t-test svm polynomial kernel and svm linear kernel", _t
