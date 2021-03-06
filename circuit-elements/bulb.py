#!/usr/bin/python
# Plots and fits data for a light bulb w/ varying potentials

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------- User-Defined Functions ----------

# Fit function for optimizing
# (x, k, a) --> (f_value)
# (float, float, float) --> (float)
def f(x, k, a):
	return k * x ** a

# ---------- Main Code ----------

# Set desired font
plt.rc('font', family = 'Times New Roman')
path_to_file = "/home/polina/Documents/3rd_Year/PHY324/circuit-elements/bulb.txt"
# Load data
V, V_unc, I, I_unc = np.loadtxt(path_to_file, unpack = True)
# In text file, potential in V, current in mA

# Use this to exclude the first n points (significantly improves reduced chi squared value)
n = 0
V = V[n:]
V_unc = V_unc[n:]
I = I[n:]
I_unc = I_unc[n:]

ddof = np.size(V) - 2	# degrees of freedom for reduced chi squared
print(ddof)

# Fit data to model function
popt, pcov = curve_fit(f, I, V, sigma = V_unc, p0 = [1, 1])
print(popt)
print "k:", popt[0], "+-", np.sqrt(pcov[0, 0])
print "a:", popt[1], "+-", np.sqrt(pcov[1, 1])
# Find residuals
r = V - f(I, *popt)
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
plt.scatter(I, V, label = "Data", s = 10, color = "black")
plt.errorbar(I, V, xerr = I_unc, yerr = V_unc, linestyle = "None", color = "black")
plt.plot(I, f(I, *popt), label = "Model", color = "black")
plt.title("Potential vs. Current of a Light Bulb")
# plt.legend()
plt.ylabel("Potential (V)")
plt.xlim([-5, 45])
plt.ylim([-4, 29])
frame1.set_xticklabels([])
# frame1.set_xticklabels([])
plt.grid(True)
# Residual plot
frame2 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
plt.scatter(I, r, s = 10, color = "black")
plt.grid(True)
plt.xlabel("Current (mA)")
plt.ylabel("Residuals")
plt.xlim([-5, 45])
plt.ylim([-0.75, 0.75])
plt.savefig("Bulb.pdf")
plt.show()
plt.close()
