import numpy as np 
from scipy.optimize import linprog 
 
def solve_simplex(): 
    # Maximize: 3x + 2y -- to Minimize: -3x - 2y 
    c = [-3, -2] 
    A = [[2, 1], [1, 1]] 
    b = [100, 80] 
    res = linprog(c, A_ub=A, b_ub=b, method='highs') 
    print("Optimal Solution:", res.x) 
    print("Optimal Objective Value:", -res.fun) 
 
if __name__ == "__main__": 
