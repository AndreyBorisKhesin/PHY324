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
temp = temp + 273.15
temp_unc = temp_unc

# Use this to exclude the first n points (significantly improves reduced chi squared value)
n = 0
temp = temp[n:]
temp_unc = temp_unc[n:]
resist = resist[n:]
resist_unc = resist_unc[n:]

ddof = np.size(temp) - 2	# degrees of freedom for reduced chi squared
print(ddof)

popt, pcov = curve_fit(f, temp, resist, sigma = resist_unc, p0 = [1e6, 10])
print(popt)
print "R_0:", popt[0], "+-", np.sqrt(pcov[0, 0])
print "T_0:", popt[1], "+-", np.sqrt(pcov[1, 1])
# Find residuals
r = resist - f(temp, *popt)
chisq = np.sum((r / resist_unc) ** 2)
print "Reduced chi squared:", chisq / ddof

fig1 = plt.figure(1)
# Plot data + model
frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
plt.scatter(temp, resist, label = "Data")
plt.errorbar(temp, resist, xerr = temp_unc, yerr = resist_unc, linestyle = "None")
plt.plot(temp, f(temp, *popt), label = "Model")
plt.title("Resistance vs. Temperature of a Thermistor")
plt.legend()
plt.ylabel("Resistance (kiloohms)")
# frame1.set_xticklabels([])
plt.grid(True)
# Residual plot
frame2 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
plt.scatter(temp, r)
plt.grid(True)
plt.xlabel("Temperature (K)")
plt.ylabel("Residuals")
plt.ylim([-2, 9])
plt.savefig("Thermistor.pdf")
plt.show()
plt.close()
