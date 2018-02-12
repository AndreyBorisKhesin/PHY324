#!/bin/python
# Linear fit of micrometer reading vs. num of fringes

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

# ---------- User-Defined Function ----------

# Linear fit function
def func(x, a, b):
	return a * x + b

# ---------- Main Code ----------

# Set desired font
plt.rc('font', family = 'Times New Roman')
path_to_file = "measurements.txt"
num, reading = np.loadtxt(path_to_file, unpack = True)

plt.plot(num, reading)
plt.show()
