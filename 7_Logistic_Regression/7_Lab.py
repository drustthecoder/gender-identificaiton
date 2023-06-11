import numpy

from mlpr import *


def func_f(x):
    y = x[0]
    z = x[1]
    return (y + 3) ** 2 + numpy.sin(y) + (z + 1) ** 2


def func_f_and_gradient(x):
    y = x[0]
    z = x[1]
    f_value = (y + 3) ** 2 + numpy.sin(y) + (z + 1) ** 2
    gradient = numpy.array([2 * (y + 3) + numpy.cos(y), 2 * (z + 1)])
    return f_value, gradient


def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
    return D, L


def sigmoid(z):
    # Sigmoid function
    return 1 / (1 + numpy.exp(-z))



if __name__ == "__main__":
    # print("func_f:")
    # x, f, d = scipy.optimize.fmin_l_bfgs_b(func_f, numpy.array([0, 0]), approx_grad=True)
    # print(f"x: {x}, f: {f}, d: {d}")
    #
    # print("func_f_and_gradient:")
    # x, f, d = scipy.optimize.fmin_l_bfgs_b(func_f_and_gradient, numpy.array([0, 0]))
    # print(f"x: {x}, f: {f}, d: {d}")

    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    # logRegModel(D, L, 1e-6).train()

    lambdas = [1e-9, 1e-6, 1e-3, 1e-1, 1]
    results = []

    for lam in lambdas:
        model = LogisticRegressionModel(DTR, LTR, DTE, LTE, lam)
        weights, loss, error_rate = model.train()
        results.append({'Lambda': lam, 'Loss': loss, 'Error Rate': error_rate})

    print("Lambda\t\tError Rate\t\tLoss")
    for result in results:
        print(f"{result['Lambda']}\t\t{round(result['Error Rate']*100, 1)}%\t\t{result['Loss']}")
