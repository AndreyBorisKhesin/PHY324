# Integrates the volume of the truncated sphere

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# ---------- User-Defined Functions ----------

# Integrad for scipy.integrate.dblquad (1)
def func1(y, x):
	return 2 * np.pi * y ** 3
	
# Lower bound for integration in y (1)
def gfun1(x):
	return 0
	
# Upper bound for integration in y (1)
# Radius R must be defined before calling the function
def hfun1(x):
	return np.sqrt(R ** 2 - x ** 2)
	
# Integrad for scipy.integrate.dblquad (2) (x component)
def func2x(y, x):
	return 2 * np.sqrt(R ** 2 - x ** 2 - y ** 2) * y
	
# Integrad for scipy.integrate.dblquad (2) (y component)
def func2y(y, x):
	return 2 * np.sqrt(R ** 2 - x ** 2 - y ** 2) * (-x)
	
# Lower bound for integration in y (2)
def gfun2(x):
	return -1 * np.sqrt(R ** 2 - x ** 2)
	
# Upper bound for integration in y (2)
# Radius R must be defined before calling the function
def hfun2(x):
	return np.sqrt(R ** 2 - x ** 2)
	
# ---------- Main Code ----------

R = 25.2625e-3		# m
delta = 50.525e-3 - 48.620e-3		# m
rho = 1
g = 1

val1 = integrate.dblquad(func1, R - delta, R, gfun1, hfun1)
print("I:", rho * (8 * np.pi * R ** 5) / 15 - val1[0], "+-", val1[1])

val2x = integrate.dblquad(func2x, - R, R - delta, gfun2, hfun2)
val2y = integrate.dblquad(func2y, - R, R - delta, gfun2, hfun2)
print("L, x:", rho * g * val2x[0], "+-", val2x[1])
print("L, y:", rho * g * val2y[0], "+-", val2y[1])
