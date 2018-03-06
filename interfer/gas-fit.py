#!/bin/python
# Linear fit of pressure vs. num of fringes passed the fied of view

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

# ---------- User-Defined Function ----------

# Linear fit function
def func(x, a):
	# return a * x + b
	return a * x

# ---------- Main Code ----------

# Set desired font
plt.rc('font', family = 'Times New Roman')
path_to_file = "gas.txt"
pressure = np.loadtxt(path_to_file, unpack = True)
num_fringes = np.arange(0, pressure.size)
pressure_unc = np.full((1, pressure.size), 5)[0]
# Degrees freedom in the fit
ddof = pressure.size - 1

# Fit the data
popt, pcov = curve_fit(func, num_fringes, pressure, sigma = pressure_unc)
print "a:", popt[0], "+-", np.sqrt(pcov[0, 0]), "(slope)"
# print "b:", popt[1], "+-", np.sqrt(pcov[1, 1])
# Calculate chi squared + residuals
r = pressure - func(num_fringes, *popt)
chisq = np.sum((r / pressure_unc) ** 2)
print "Reduced chi squared:", chisq / ddof
# Calculate and print R^2
ss_res = np.sum(r ** 2)
ss_tot = np.sum((pressure - np.mean(pressure)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print "R^2:", r_squared

# Plot micrometer reading vs. carriage displacement
fig1 = plt.figure(1)
frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
plt.scatter(num_fringes, pressure, color = "black", s = 10)
plt.plot(num_fringes, func(num_fringes, *popt), color = "black")
plt.grid(True)
plt.ylabel("Pressure (mm Hg)")
plt.xlim([-2, 48])
frame1.set_xticklabels([])
plt.title("Pressure vs. number of fringes passing the field of view")
frame2 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
plt.scatter(num_fringes, r, color = "black", s = 10)
plt.ylabel("Residuals")
plt.xlabel("Number of fringes")
plt.xlim([-2, 48])
plt.grid(True)
plt.savefig("gas-fit.pdf")
plt.show()

