import math
import numpy
from sympy import *
from autograd import grad

function = lambda x1, x2: x1 ** 2 + 7 * x2 ** 2 - x1 * x2 + x1
calc_t = lambda x1, x2: ((5 * x1 ** 2 - 32 * x1 * x2 + 4 * x1 + 197 * x2 ** 2 - 2 * x1 + 1) /
                         (26 * x1 ** 2 - 458 * x1 * x2 + 10 * x1 + 2774 * x2 ** 2 - 32 * x2 + 2))
x0 = numpy.array([1.1, 1.1])
eps1 = 0.1
eps2 = 0.15
iterCounts = 10


def norm(args):
    return math.sqrt(sum(map(lambda x: x ** 2, args)))


def is_positive_def_matrix(matrix):
    return numpy.all(numpy.linalg.eigvals(matrix) > 0)


def newtons_method(epsilon_1, epsilon_2, max_iter_counts, init_approximation, func, calc_step):
    x_min = numpy.array([])
    gradient = grad(func, [0, 1])
    x1, x2 = symbols("x1, x2")
    hessian_matrix = [[lambdify([x1, x2], diff(func(x1, x2), x1, x1)),
                       lambdify([x1, x2], diff(func(x1, x2), x1, x2))],
                      [lambdify([x1, x2], diff(func(x1, x2), x2, x1)),
                       lambdify([x1, x2], diff(func(x1, x2), x2, x2))]]
    iter_counts = 0
    check_counts = 0
    new_approximation = init_approximation.copy()
    while True:
        iter_counts += 1
        gradient_in_point = gradient(new_approximation[0], new_approximation[1])
        gradient_norm = norm(gradient_in_point)
        if gradient_norm < epsilon_1:
            x_min = new_approximation
            break
        if iter_counts >= max_iter_counts:
            x_min = new_approximation
            break
        hessian_matrix_in_point = numpy.array([
            [hessian_matrix[0][0](new_approximation[0], new_approximation[1]),
             hessian_matrix[0][1](new_approximation[0], new_approximation[1])],
            [hessian_matrix[1][0](new_approximation[0], new_approximation[1]),
             hessian_matrix[1][1](new_approximation[0], new_approximation[1])]])
        inverse_hessian_matrix = numpy.linalg.inv(hessian_matrix_in_point)
        if is_positive_def_matrix(inverse_hessian_matrix):
            d = -1 * inverse_hessian_matrix @ gradient_in_point
            t = 1
        else:
            d = -1 * gradient_in_point
            t = calc_step(new_approximation[0], new_approximation[1])
        prev_approximation = new_approximation.copy()
        new_approximation = new_approximation + t * d
        if (norm(new_approximation - prev_approximation) < epsilon_2 and abs(
                func(new_approximation[0], new_approximation[1]) - func(prev_approximation[0], prev_approximation[1]))):
            if check_counts == 2:
                x_min = new_approximation
                break
            else:
                check_counts += 1
    return {"x_min": list(x_min), "f(x_min)": func(x_min[0], x_min[1]), "iterations": iter_counts}


print(newtons_method(eps1, eps2, iterCounts, x0, function, calc_t))
