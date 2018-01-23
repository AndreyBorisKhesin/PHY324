#!/usr/bin/python
# Calibration curve of the thermocouple; estimates Seebeck constant using curve_fit

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------- User-Defined Functions ----------

# Fit function for optimizing; a  = S (Seebeck constant)
# (x, a) --> (f_value)
# (float, float) --> (float)
def f(x, a):
	return a * x

# ---------- Main Code ----------

# Set desired font
plt.rc('font', family = 'Times New Roman')

path_to_file = "/home/polina/Documents/3rd_Year/PHY324/thermocouple/measurements.txt"
# Load data
T1, T1_unc, T2, T2_unc, V, V_unc = np.loadtxt(path_to_file, unpack = True)
# Convert data to Kelvin
T1 = T1 + 273.15
T2 = T2 + 273.15
T_diff = T2 - T1
T_diff_unc = np.sqrt(T1_unc ** 2 + T2_unc ** 2)
print(T_diff_unc)

ddof = np.size(T_diff) - 1	# degrees of freedom for reduced chi squared
print ddof

# Fit data to model function
popt, pcov = curve_fit(f, T_diff, V, sigma = V_unc)
print(popt)
print "S:", popt[0], "+-", np.sqrt(pcov[0, 0])
# Find residuals
r = V - f(T_diff, *popt)
chisq = np.sum((r / V_unc) ** 2)
print "Reduced chi squared:", chisq / ddof

# Calculate and print R^2
ss_res = np.sum(r ** 2)
ss_tot = np.sum((V - np.mean(V)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print "R^2:", r_squared

fig1 = plt.figure(1)
# Plot data + model
frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
plt.scatter(T_diff, V, s = 3, color = "black")
plt.errorbar(T_diff, V, xerr = T_diff_unc, yerr = V_unc, linestyle = "None", color = "black", linewidth = 0.5)
plt.plot(T_diff, f(T_diff, *popt), color = "black", linewidth = 0.5)
plt.title("Potential vs. temperature difference of a thermocouple")
plt.xlim([-5, 105])
plt.ylim([-0.5, 4.5])
plt.ylabel("EMF (mV)")
frame1.set_xticklabels([])
plt.grid(True)
# Residual plot
frame2 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
plt.scatter(T_diff, r, s = 3, color = "black")
plt.xlim([-5, 105])
plt.ylim([-0.19, 0.19])
plt.xlabel("Temperature difference (K)")
plt.ylabel("Residuals")
plt.grid(True)
plt.savefig("ThermocoupleCalibration.pdf")
plt.show()

