#!/usr/bin/python
# Plots and fits data for a thermistor w/ varying temperatures

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------- User-Defined Functions ----------

# Fit function for optimizing; a = R_0, b = T_0
# (x, a, b) --> (f_value)
# (float, float, float) --> (float)
def f(x, a, b):
	return a * np.exp(- x / b)

# ---------- Main Code ----------

path_to_file = "/home/polina/Documents/3rd_Year/PHY324/circuit-elements/thermistor.txt"
# Load data
temp, temp_unc, resist, resist_unc = np.loadtxt(path_to_file, unpack = True)
# Convert data to Kelvin, kiloohms
temp = temp + 273.0
temp_unc = temp_unc + 273.0
ddof = np.size(temp) - 2	# degrees of freedom for reduced chi squared
print(ddof)

popt, pcov = curve_fit(f, temp, resist, sigma = resist_unc, p0 = [1e6, 10])
print(popt)
# Find residuals
r = temp - f(temp, *popt)
chisq = np.sum((r / resist_unc) ** 2)
print "Reduced chi squared:", chisq / ddof

plt.plot(temp, resist)
plt.plot(temp, f(temp, *popt))
plt.grid(True)
plt.xlabel("Temperature (K)")
plt.ylabel("Resistance (kiloohms)")
plt.title("Resistance vs. Temperature for a Thermistor")
plt.show()
