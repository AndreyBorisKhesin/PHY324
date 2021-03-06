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

# Set desired font
plt.rc('font', family = 'Times New Roman')
path_to_file = "/home/polina/Documents/3rd_Year/PHY324/circuit-elements/thermistor.txt"
# Load data
temp, temp_unc, resist, resist_unc = np.loadtxt(path_to_file, unpack = True)
# Convert data to Kelvin, kiloohms
temp = temp + 273.15
temp_unc = temp_unc

# Use this to exclude the last n points (significantly improves reduced chi squared value)
n = 0
temp = temp[:np.size(temp) - n]
temp_unc = temp_unc[:np.size(temp_unc) - n]
resist = resist[:np.size(resist) - n]
resist_unc = resist_unc[:np.size(resist_unc) - n]

ddof = np.size(temp) - 2	# degrees of freedom for reduced chi squared

# Fit data to model function
popt, pcov = curve_fit(f, temp, resist, sigma = resist_unc, p0 = [1e6, 10])
print(popt)
print "R_0:", popt[0], "+-", np.sqrt(pcov[0, 0])
print "T_0:", popt[1], "+-", np.sqrt(pcov[1, 1])
# Find residuals
r = resist - f(temp, *popt)
chisq = np.sum((r / resist_unc) ** 2)
print "Reduced chi squared:", chisq / ddof

# Calculate and print R^2
ss_res = np.sum(r ** 2)
ss_tot = np.sum((resist - np.mean(resist)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print "R^2:", r_squared

fig1 = plt.figure(1)
# Plot data + model
frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
plt.scatter(temp, resist, label = "Data", s = 10, color = "black")
plt.errorbar(temp, resist, xerr = temp_unc, yerr = resist_unc, linestyle = "None", color = "black")
plt.plot(temp, f(temp, *popt), label = "Model", color = "black")
plt.title("Resistance vs. Temperature of a Thermistor")
# plt.legend()
plt.ylabel("Resistance (kiloohms)")
plt.xlim([270, 360])
# frame1.axes().get_xaxis().set_visible(False)
frame1.set_xticklabels([])
plt.grid(True)
# Residual plot
frame2 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
plt.scatter(temp, r, s = 10, color = "black")
plt.grid(True)
plt.xlabel("Temperature (K)")
plt.ylabel("Residuals")
plt.xlim([270, 360])
plt.ylim([-3, 9])
plt.savefig("Thermistor.pdf")
plt.show()
plt.close()
