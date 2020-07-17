#!/usr/bin/python

import sys
sys.path.insert(1, 'libsvm/python')
import os
import svmutil


def clear_folds(self):
    for k in range(0, 10):
        folder = 'fold_' + str(k) + '/'
        files = os.listdir(folder)
        files_to_remove = list(filter(lambda x: x != self.filename + '.data'
                                      and x != self.filename + '.data.svm'
                                      and x != self.filename + '.test'
                                      and x != self.filename + '.test.svm'
                                      and x != self.filename + '.nb'
                                      and x != self.filename + '.names', files))
        for f in files_to_remove:
            file_path = folder + f
            self.print_debug("removing", file_path)
            os.remove(file_path)


def print_debug(self, arg1, *argv):
    if self.debug:
        print arg1
        for arg in argv:
            print arg


class SvmLib:

    def __init__(self, filename, debug=False):
        self.filename = filename
        self.debug = debug

    def print_debug(self, arg1, *argv):
        print_debug(self, arg1, *argv)

    # Esta funcion en la encargada de generar el modelo con los parametros correspondientes
    def train_fold(self, k, c):
        self.print_debug('train_fold', k, c)
        folder_name = 'fold_' + str(k) + '/'
        file_name = self.filename + '.data.svm'
        os.chdir(folder_name)
        if os.path.exists(file_name + '.model'):
            os.remove(file_name + '.model')
        command = '../libsvm/svm-train -s 0 -t 0 -c ' + str(c) \
                  + ' ' + file_name + ' > /dev/null'
        self.print_debug('train_fold', command)
        os.system(command)
        os.chdir('../')

    def train_fold_polynomial(self, k, c, g, d):
        self.print_debug('train_fold_polynomial', k, c, g, d)
        folder_name = 'fold_' + str(k) + '/'
        file_name = self.filename + '.data.svm'
        os.chdir(folder_name)
        if os.path.exists(file_name + '.model'):
            self.print_debug("deleting", file_name + '.model')
            os.remove(file_name + '.model')
        command = '../libsvm/svm-train -s 0 -t 1 -c ' + str(c) \
                  + ' -g ' + str(g) + ' -d ' + str(d) \
                  + ' ' + file_name + ' > /dev/null'
        self.print_debug('train_fold_polynomial', command)
        os.system(command)
        os.chdir('../')

    # Testea el modelo en la fold correspondiente (siempre es la 9)
    def testing_fold(self, k):
        folder_name = 'fold_' + str(k) + '/'
        os.chdir(folder_name)
        if os.path.exists('output.txt'):
            os.remove('output.txt')
        testing_command = '../libsvm/svm-predict ' + self.filename + '.test.svm ' \
                          + self.filename + '.data.svm.model ../fold_' + str(k) + '/' \
                          + self.filename + '.svm.predict | grep Accuracy > output.txt'
        self.print_debug(testing_command)
        os.system(testing_command)
        line = open('output.txt').read()
        self.print_debug(line)
        os.chdir('../')
        return float("{:.2f}".format(float(line.split(' ')[2].replace('%', ''))))

    def fit_linear_parameters(self):
        max_accuracy = -1.
        selected_i = 0
        selected_c = 0
        for i in range(-5, 5):
            self.print_debug(str(i), str(pow(10, i)))
            self.train_fold(0, pow(10, i))
            acc = self.testing_fold(0)
            if acc > max_accuracy:
                selected_i = i
                selected_c = pow(10, i)
                max_accuracy = acc

        if selected_i >= 0:
            range_i = pow(10, selected_i + 1) - pow(10, selected_i)
        else:
            range_i = pow(10, selected_i) - pow(10, selected_i - 1)

        step_i = range_i / 10.

        self.print_debug("SELECTED C", selected_c)
        for s in range(1, 10):
            c = selected_c + s * step_i
            self.print_debug(c)
            self.train_fold(0, c)
            acc = self.testing_fold(0)
            if acc > max_accuracy:
                selected_c = c
                max_accuracy = acc

        self.print_debug("SELECTED C", selected_c, max_accuracy)
        return selected_c

    def fit_polynomial_parameters(self):
        max_accuracy = -1.
        selected_i_c = 0
        temp_selected_c = 0
        selected_j_g = 0
        temp_selected_g = 0
        selected_d = 0
        temp_selected_d = 2
        for i in range(-2, 2):
            for j in range(-2, 2):
                for d in range(2, 11, 4):
                    c = pow(10, i)
                    g = pow(10, j)
                    self.train_fold_polynomial(0, c, g, d)
                    acc = self.testing_fold(0)
                    if acc > max_accuracy:
                        selected_i_c = i
                        temp_selected_c = c
                        selected_j_g = j
                        temp_selected_g = g
                        temp_selected_d = d
                        max_accuracy = acc

        selected_c = temp_selected_c
        selected_g = temp_selected_g
        selected_d = temp_selected_d
        error = 1. - (max_accuracy / 100)
        self.print_debug("ERROR", error)
        #calculo si los valores entre 1 y 9 superan al de 0
        # (ACLARAR BIEN)
        self.print_debug("SELECTED C first iterations", temp_selected_c, temp_selected_g)
        for i in range(1, 10):
            temp_selected_c = temp_selected_c + temp_selected_c * error
            temp_selected_g = temp_selected_g + temp_selected_g * error
            if temp_selected_d < 10:
                for h in range(1, 4):
                    d = temp_selected_d + h
                    self.print_debug(temp_selected_c, temp_selected_g, d)
                    self.train_fold_polynomial(0, temp_selected_c, temp_selected_c, d)
                    acc = self.testing_fold(0)
                    if acc > max_accuracy:
                        selected_c = temp_selected_c
                        selected_g = temp_selected_g
                        selected_d = d
                        max_accuracy = acc
            else:
                self.print_debug(temp_selected_c, temp_selected_g, selected_d)
                self.train_fold_polynomial(0, temp_selected_c, temp_selected_g, selected_d)
                acc = self.testing_fold(0)
                if acc > max_accuracy:
                    selected_c = temp_selected_c
                    selected_g = temp_selected_g
                    max_accuracy = acc

        self.print_debug("SELECTED C", selected_c, max_accuracy)
        return {"c": selected_c, "d": selected_d, "g": selected_g}

    def test_model(self, c):
        errors = []
        for k in range(0, 10):
            self.train_fold(k, c)
            acc = self.testing_fold(k)
            error = 1 - acc / 100
            self.print_debug(k, acc)
            errors.append(float("{:.2f}".format(error)))

        self.print_debug('model tested with accuracies', errors)
        return errors

    def test_polynomial_model(self, c, g, d):
        errors = []
        for k in range(0, 10):
            self.train_fold_polynomial(k, c, g, d)
            acc = self.testing_fold(k)
            error = 1 - acc / 100
            self.print_debug(k, acc)
            errors.append(float("{:.2f}".format(error)))

        self.print_debug('model tested with accuracies', errors)
        return errors

    def clear_folds(self):
        clear_folds(self)


class DTreeLib:

    def __init__(self, filename, debug=False):
        self.filename = filename
        self.debug = debug

    def train_and_test(self):
        test_errors = []
        for k in range(0, 10):
            folder_name = 'fold_' + str(k) + '/'
            os.chdir(folder_name)
            command = '../c4.5 -f ' + self.filename + " -u | grep '<<' > output.txt"
            os.system(command)
            line = open('output.txt').read()
            self.print_debug(line)
            line_split = line.split("<<")
            _training_line = line_split[0].split("  ")
            _test_line = line_split[1].split("  ")

            training_line = filter(lambda x:
                                   x != '' and x != ' ' and '\n' not in x and '\t' not in x,
                                   _training_line)

            test_line = filter(lambda x:
                               x != '' and x != ' ' and '\n' not in x and '\t' not in x,
                               _test_line)
            training_error = float(training_line[3].split('(')[1].replace("%)", "")) / 100
            self.print_debug(training_line)
            self.print_debug(test_line)
            test_error = float(test_line[3].split('(')[1].replace("%)", "")) / 100
            self.print_debug(training_error, test_error)
            test_errors.append(float("{:.2f}".format(test_error)))
            os.chdir('../')
        return test_errors

    def clear_folds(self):
        clear_folds(self)

    def print_debug(self, arg1, *argv):
        print_debug(self, arg1, *argv)


class NaiveBayesLib:

    def __init__(self, filename, debug = False):
        self.filename = filename
        self.debug = debug

    def clear_folds(self):
        clear_folds(self)

    def print_debug(self, arg1, *argv):
        print_debug(self, arg1, *argv)

    def train_and_test(self):
        errors = []
        for k in range(0, 10):
            folder_name = 'fold_' + str(k) + '/'
            os.chdir(folder_name)
            command = '../nb_n ' + self.filename + " | grep Test > output.txt"
            os.system(command)
            line = open('output.txt').read()
            self.print_debug(line)
            line_splitted = line.split(':')[1]
            test_error = float(line_splitted.replace('%', ''))
            errors.append(test_error / 100)
            os.chdir('../')
        return errors

