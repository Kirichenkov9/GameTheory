import math
import numpy as np
import fractions
from sympy import Symbol
import warnings
warnings.filterwarnings('ignore')

def get_row_by_index(matrix, index):
    return matrix[index]

def get_column_by_index(matrix, index):
    return [matrix[i][index] for i in range(len(matrix))]

def vector_addition(a, b):
    return [i + j for i, j in zip(a, b)]

def brown_robinson_method(matrix, eps):
    m = len(matrix)    
    n = len(matrix[0]) 

    x = m * [0]
    y = n * [0]

    curr_strategy_a = 0
    curr_strategy_b = 0

    win_a = m * [0]
    loss_b = n * [0]
    curr_eps = math.inf
    k = 0

    lower_bounds = []
    upper_bounds = []

    while (curr_eps > eps):
        k += 1
        win_a = vector_addition(win_a, get_column_by_index(matrix, curr_strategy_b))
        loss_b = vector_addition(loss_b, get_row_by_index(matrix, curr_strategy_a))
        x[curr_strategy_a] += 1
        y[curr_strategy_b] += 1

        lower_bound = min(loss_b) / k
        upper_bound = max(win_a) / k
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

        curr_eps = min(upper_bounds) - max(lower_bounds)
        
        curr_strategy_a = np.argmax(win_a)
        curr_strategy_b = np.argmin(loss_b)

    cost = max(lower_bounds) + curr_eps / 2

    x = [i / k for i in x]
    y = [i / k for i in y]

    return x, y, cost

def kernel_function(x, y, a, b, c, d, e):
    return a * x**2 + b * y**2 + c * x * y + d * x + e * y

def find_saddle_point(mat):
    max_loss = np.amax(mat, axis=0)
    min_max = np.amin(max_loss)
    y = np.argmin(max_loss)
    
    min_win = np.amin(mat, axis=1)
    max_min = np.amax(min_win)
    x = np.argmax(min_win)

    return max_min if max_min == min_max else 0, x, y

def average(a):
    return sum(a) / len(a)

def limit(a, eps):
    N = -1
    ff = False
    for i in range(0, len(a) - 1):
        ff = True
        for j in range(i + 1, len(a)):
            if abs(a[j] - a[i]) >= eps:
                ff = False
                break
        if ff:
            N = i
            break
    if not ff:
        return math.inf
    return average([min(a[N + 1: ]), max(a[N + 1: ])])

def generate_grid_approximation(n, a, b, c, d, e):
    return [[kernel_function(i / n, j / n, a, b, c, d, e) for j in range(n + 1)] for i in range(n + 1)]

def grid_approximation_method(eps, a, b, c, d, e):
    cost_array = []
    x_array = []
    y_array = []
    n = 1
    while True:
        cur_H, x, y, h, saddle_point = approximation_method_step(eps, n, a, b, c, d, e)
        cost_array.append(h)
        lim = limit(cost_array, eps)
        if lim != math.inf:
            x_array.append(x)
            y_array.append(y)

        stop_lim = limit(cost_array, fractions.Fraction(eps, 10))
        if stop_lim != math.inf:
            print(f"Found solution on {n} iteration:")
            print("x = {:.3f}, y = {:.3f}, h = {:.3f}".format(float(average(x_array)), float(average(y_array)), float(lim)))
            return average(x_array), average(y_array), lim
           
        print_result(cur_H, n, x, y, h, saddle_point, eps)
        n += 1

def approximation_method_step(eps, n, a, b, c, d, e):
    cur_H = generate_grid_approximation(n, a, b, c, d, e)
    
    saddle_point, x, y = find_saddle_point(np.asarray(cur_H))
    if saddle_point:
        h = saddle_point
        x = fractions.Fraction(x, n)
        y = fractions.Fraction(y, n)
    else:
        x, y, h = brown_robinson_method(cur_H, eps)
        x = fractions.Fraction(np.argmax(x), n)
        y = fractions.Fraction(np.argmax(y), n)
    return cur_H, x, y, h, saddle_point

def print_result(H,n,x, y, h, saddle_point, eps):
    print(f"N = {n}")
    if n <= 10:
        for i in H:
            print(*["{:8.3f}".format(float(j)) for j in i])
    
    if saddle_point:
        print("Has saddle point\nx = {:}, y = {:}, h = {:.3f}".format(x, y, float(saddle_point)))
    else:
        print("Hasn't saddle point")
        print("Calculated with Brown-Robinson method with accuracy eps = {:.3f}\nx = {:}, y = {:}, h = = {:.3f}".format(float(eps), x, y, float(h)))

def ask_user():
    check = str(input("Load conditions from file? (Y/N): ")).lower().strip()
    try:
        if check[0] == 'y':
            return True
        elif check[0] == 'n':
            return False
        else:
            print('Invalid Input')
            return ask_user()
    except Exception as error:
        print("Please enter valid inputs")
        print(error)
        return ask_user()

def get_conditions_file():
    filename = str(input("Enter filename: "))
    lines = []

    with open(filename, 'r') as file:
        lines = file.readlines()
    try:
        a = fractions.Fraction(lines[0])
        b = fractions.Fraction(lines[1])
        c = fractions.Fraction(lines[2])
        d = fractions.Fraction(lines[3])
        e = fractions.Fraction(lines[4])
    except ValueError as err:
        print(f"Incorrect values: {err}")
    return a, b, c, d, e

def get_condiditions_user_input():
    try:
        a = fractions.Fraction(input("a >> "))
        b = fractions.Fraction(input("b >> "))
        c = fractions.Fraction(input("c >> "))
        d = fractions.Fraction(input("d >> "))
        e = fractions.Fraction(input("e >> "))
    except ValueError as err:
        print(f"Incorrect values: {err}")
    return a, b, c, d, e

def get_conditions():
    return get_conditions_file() if ask_user() else get_condiditions_user_input()

def analytical_method(a, b, c, d, e):
    x = Symbol('x')
    y = Symbol('y')

    _H = a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y

    print('Derivate of H = ', _H)
    Hxx = _H.diff(x, 2)
    Hyy = _H.diff(y, 2)
    print('Derivate Hxx = ', Hxx)
    print('Derivate Hyy = ', Hyy)

    if float(Hxx) < 0 and float(Hyy) > 0:
        print("The game is convex-concave")
    else:
        print("The game isn't convex-concave")

    Hx = _H.diff(x)
    Hy = _H.diff(y)

    print('Derivate Hx = ', Hx)
    print('Derivate Hy = ', Hy)

    y_sol = (c * d - 2 * a * e) / (4 * b * a - c * c)
    x_sol = -(c * y_sol + d) / (2 * a)
    h = kernel_function(float(x_sol), float(y_sol), a, b, c, d, e)
    print("x = {:}, y = {:}, h = {:.3f}".format(x_sol, y_sol, h))

def main():
    p = 3
    a, b, c, d, e = get_conditions()
    grid_approximation_method(fractions.Fraction(1, 10**p), a, b, c, d, e)
    print("Analytical method")
    analytical_method(a, b, c, d, e)
    
if __name__ == "__main__":
    main()
