#!/usr/bin/python

import sys
sys.path.insert(1, 'libsvm/python')
import os
import svmutil


#Esta funcion limpia las carpetas de datos que puedan haber quedado
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


# Esta funcion imprime en pantalla, siempre y cuando debug este habilitado
def print_debug(self, arg1, *argv):
    if self.debug:
        print "----"
        print arg1
        for arg in argv:
            print arg
        print "----"


class SvmLib:

    def __init__(self, filename, debug=False, iterations=10, gamma=1.):
        self.filename = filename
        self.debug = debug
        self.iterations = iterations
        self.gamma = gamma

    def print_debug(self, arg1, *argv):
        print_debug(self, arg1, *argv)

    # Esta funcion en la encargada de generar el modelo con los parametros correspondientes
    # usando un kernel lineal (o bien podriamos decir, sin usar kernel)
    def train_fold(self, k, c):
        self.print_debug('train_fold', k, c)
        folder_name = 'fold_' + str(k) + '/'
        file_name = self.filename + '.data.svm'
        y, x = svmutil.svm_read_problem(folder_name + file_name)
        prob = svmutil.svm_problem(y, x)
        param = svmutil.svm_parameter('-s 0 -t 0 -c ' + str(c))
        m = svmutil.svm_train(prob, param)
        return m

    # Esta funcion en la encargada de generar el modelo con los parametros correspondientes
    # usando un kernel polinomial. K es el la fold que sera usada para testing
    # c, g (gamma) y d (degree) son los parametros para entrenar el modelo
    def train_fold_polynomial(self, k, c, g, d):
        self.print_debug('train_fold_polynomial', k, c, g, d)
        folder_name = 'fold_' + str(k) + '/'
        file_name = self.filename + '.data.svm'
        y, x = svmutil.svm_read_problem(folder_name + file_name)
        prob = svmutil.svm_problem(y, x, isKernel=True)
        param = svmutil.svm_parameter('-s 0 -t 1 -c ' + str(c)
                                      + ' -g ' + str(g) + ' -d ' + str(d))
        m = svmutil.svm_train(prob, param)
        return m

    # Testea el modelo en la fold k usando el modelo m
    def testing_fold(self, k, m):
        folder_name = 'fold_' + str(k) + '/'
        file_name = self.filename + '.test.svm'
        y, x = svmutil.svm_read_problem(folder_name + file_name)
        print "Y", len(y), "X",  len(x)
        p_label, p_acc, p_val = svmutil.svm_predict(y, x, m)
        return float(p_acc[0])

    # Ajusta el parametro C para svm con kernel lineal.
    # Primero itera sobre potencias de 10. Luego, realiza un ajuste mas
    # fino sobre la zona obtenida.
    def fit_linear_parameters(self):
        max_accuracy = -1.
        selected_c = 1
        selected_i = 0
        for i in range(-5, 5):
            self.print_debug(str(i), str(pow(10, i)))
            m = self.train_fold(0, pow(10, i))
            acc = self.testing_fold(0, m)
            if acc > max_accuracy:
                selected_c = pow(10, i)
                selected_i = i
                max_accuracy = acc
        error = 1. - (max_accuracy/100)

        diff = selected_c - pow(10, selected_i - 1)

        self.print_debug("SELECTED C", selected_c, error)
        c = selected_c - diff / 2
        for i in range(0, self.iterations):
            c = c + c * error * self.gamma
            self.print_debug(c)
            m = self.train_fold(0, c)
            acc = self.testing_fold(0, m)
            error = 1. - (acc/100)
            self.print_debug("NEW ERROR " + str(error) + " " + str(acc))
            if acc > max_accuracy:
                selected_c = c
                max_accuracy = acc

        self.print_debug("SELECTED C", selected_c, max_accuracy)
        return selected_c

    # Ajusta el parametro C, g y d para svm con kernel polinomial.
    # Primero itera sobre potencias de 10. Luego, realiza un ajuste mas
    # fino sobre la zona obtenida. Para el grado del polinomio, lo hace
    # entre 2 y 10. Para valores muy altos puede superar el numero de
    # iteraciones de la libreria
    def fit_polynomial_parameters(self):
        max_accuracy = -1.
        temp_selected_c = 0
        temp_selected_g = 0
        temp_selected_d = 2
        selected_i = 0
        selected_j = 0
        for i in range(-2, 2):
            for j in range(-2, 2):
                for d in range(2, 11, 4):
                    c = pow(10, i)
                    g = pow(10, j)
                    m = self.train_fold_polynomial(0, c, g, d)
                    acc = self.testing_fold(0, m)
                    if acc > max_accuracy:
                        temp_selected_c = c
                        temp_selected_g = g
                        temp_selected_d = d
                        selected_i = i
                        selected_j = j
                        max_accuracy = acc

        diff_c = temp_selected_c - pow(10, selected_i-1)
        diff_g = temp_selected_c - pow(10, selected_j - 1)

        selected_c = temp_selected_c
        selected_g = temp_selected_g
        selected_d = temp_selected_d
        temp_selected_c = temp_selected_c - (diff_c / 2)
        temp_selected_g = temp_selected_g - (diff_g / 2)
        error = 1. - (max_accuracy / 100)
        self.print_debug("ERROR", error)
        self.print_debug("SELECTED C first iterations", temp_selected_c, temp_selected_g, selected_c, selected_g)
        for i in range(0, self.iterations):
            temp_selected_c = temp_selected_c + temp_selected_c * error * self.gamma
            temp_selected_g = temp_selected_g + temp_selected_g * error * self.gamma
            if temp_selected_d < 10:
                for h in range(0, 4):
                    d = temp_selected_d + h
                    self.print_debug(temp_selected_c, temp_selected_g, d)
                    m = self.train_fold_polynomial(0, temp_selected_c, temp_selected_c, d)
                    acc = self.testing_fold(0, m)
                    if acc > max_accuracy:
                        selected_c = temp_selected_c
                        selected_g = temp_selected_g
                        selected_d = d
                        max_accuracy = acc
            else:
                self.print_debug(temp_selected_c, temp_selected_g, selected_d)
                m = self.train_fold_polynomial(0, temp_selected_c, temp_selected_g, selected_d)
                acc = self.testing_fold(0, m)
                if acc > max_accuracy:
                    selected_c = temp_selected_c
                    selected_g = temp_selected_g
                    max_accuracy = acc
            error = 1. - (acc / 100)

        self.print_debug("SELECTED C", selected_c, max_accuracy)
        return {"c": selected_c, "d": selected_d, "g": selected_g}

    # esta funcion testea un modelo svn con kernel lineal. Entrena el
    # modelo usando parametro c y luego lo testea. Repite el proceso
    # para las 10 folds
    def test_model(self, c):
        errors = []
        for k in range(0, 10):
            m = self.train_fold(k, c)
            acc = self.testing_fold(k, m)
            error = 1 - acc / 100
            self.print_debug(k, acc)
            errors.append(float("{:.4f}".format(error)))

        self.print_debug('model tested with accuracies', errors)
        return errors

    # idem al anterior, pero para svm con kernel polinomial
    def test_polynomial_model(self, c, g, d):
        errors = []
        for k in range(0, 10):
            m = self.train_fold_polynomial(k, c, g, d)
            acc = self.testing_fold(k, m)
            error = 1 - acc / 100
            self.print_debug(k, acc)
            errors.append(float("{:.4f}".format(error)))

        self.print_debug('model tested with accuracies', errors)
        return errors

    def clear_folds(self):
        clear_folds(self)


class DTreeLib:

    def __init__(self, filename, debug=False):
        self.filename = filename
        self.debug = debug

    # entrena y testea usando los patrones de las 10 folds usando
    # c4.5
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

    # entrena y testea usando los patrones de las 10 folds usando
    # naive-bayes
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

