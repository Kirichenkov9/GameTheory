import math
import fractions
import random
import numpy as np
from prettytable import PrettyTable
import warnings
warnings.filterwarnings('ignore')

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
        eps = float(lines[0])
        rows_number = int(lines[1])
        column_number = int(lines[2])
        entries = list(map(int, lines[3].split()))

        matrix = np.array(entries).reshape(rows_number, column_number)
    except ValueError as e:
        print(f"Incorrect values: {e}")
    return eps, matrix

def get_condiditions_user_input():
    try:
        eps = float(input("Enter eps: "))
        rows_number = int(input("Enter the number of rows:"))
        column_number = int(input("Enter the number of columns:"))
  
        print("Enter the entries in a single line (separated by space): ")

        entries = list(map(int, input().split()))

        matrix = np.array(entries).reshape(rows_number, column_number)
    except ValueError as e:
        print(f"Incorrect values: {e}")
    return eps, matrix

def get_conditions():
    return get_conditions_file() if ask_user() else get_condiditions_user_input()

def get_row_by_index(matrix, index):
    return matrix[index]

def get_column_by_index(matrix, index):
    return [matrix[i][index] for i in range(len(matrix))]

def get_max_index(arr):
    result = np.where(arr == np.amax(arr))[0] 
    return result[0] if len(result) == 1 else random.choice(result)

def get_min_index(arr):
    result = np.where(arr == np.amin(arr))[0]
    return result[0] if len(result) == 1 else random.choice(result)

def vector_addition(a, b):
    return [i + j for i, j in zip(a, b)]

def brown_robinson_method(matrix, eps, table):
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

    while curr_eps > eps:
        k += 1
        win_a = vector_addition(win_a, get_column_by_index(matrix, curr_strategy_b))
        loss_b = vector_addition(loss_b, get_row_by_index(matrix, curr_strategy_a))
        x[curr_strategy_a] += 1
        y[curr_strategy_b] += 1

        lower_bound = fractions.Fraction(min(loss_b), k)
        upper_bound = fractions.Fraction(max(win_a), k)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

        curr_eps = min(upper_bounds) - max(lower_bounds)
        
        table.add_row([k, curr_strategy_a + 1, curr_strategy_b + 1, *win_a, *loss_b, upper_bound, lower_bound, curr_eps])

        curr_strategy_a = get_max_index(win_a)
        curr_strategy_b = get_min_index(loss_b)

    cost = max(lower_bounds) + fractions.Fraction(curr_eps, 2)

    x = [fractions.Fraction(i, k) for i in x]
    y = [fractions.Fraction(i, k) for i in y]

    return x, y, cost

def analytical_method(matrix):
    c_inv = np.linalg.inv(matrix)
    u = np.array([[1 for _ in range(len(matrix))]])
    u_t = u.T

    cost = 1 / np.dot(np.dot(u,c_inv),u_t)

    x = c_inv.dot(u_t) * cost
    y = u.dot(c_inv) * cost

    cost = cost[0][0]

    x = [i[0] for i in x]
    y = [i for i in y][0]
    return x, y, cost

def main():
    table = PrettyTable()
    table.field_names = ["k", "A", "B", "x1", "x2", "x3", "y1", "y2", "y3", "UpBound", "LowBound", "EPS"]
    eps, matrix = get_conditions()

    print("Analytical method")
    x, y, cost = analytical_method(matrix)
    print(f"x = ", *x)
    print(f"y = ", *y)
    print("Cost = {:}, {:.3f}".format(cost, float(cost)))

    print("Brown-Robinson method")
    x, y, cost = brown_robinson_method(matrix, eps, table)
    print(table)
    print("x = (",*x, ")")
    print("y = (",*y, ")")
    print("Cost = {:}, {:.3f}".format(cost, float(cost)))

if __name__ == '__main__':
    main()
