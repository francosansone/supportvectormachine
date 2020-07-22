#!/usr/bin/python

import math
import statistics

decision_tree = [0.19607843, 0.13725491, 0.23529412, 0.15686275, 0.31999999, 0.16, 0.20408163999999998, 0.26530612000000003, 0.12244897999999999, 0.18367346999999998]

naive_bayes = [0.12, 0.2, 0.18, 0.23, 0.3, 0.18, 0.31, 0.37, 0.18, 0.24]

svm_linear = [0.16, 0.18, 0.2, 0.18, 0.3, 0.14, 0.22, 0.33, 0.12, 0.16]

diffs = [naive_bayes[i] - decision_tree[i] for i in range(0, 10)]

d = statistics.mean(diffs)

s_d = statistics.stdev(diffs) / math.sqrt(10)

t = d/s_d

print "t-test decision tree and naive bayes", t

_diffs = [naive_bayes[i] - svm_linear[i] for i in range(0, 10)]
 
_d = statistics.mean(_diffs)

_s_d = statistics.stdev(_diffs) / math.sqrt(10)

_t = _d / _s_d

print "t-test naive bayes and svm linear kernel", _t
