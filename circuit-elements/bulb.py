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

popt, pcov = curve_fit(f, I, V, sigma = V_unc, p0 = [1, 1])
print(popt)
print "k:", popt[0], "+-", np.sqrt(pcov[0, 0])
print "a:", popt[1], "+-", np.sqrt(pcov[1, 1])
# Find residuals
r = V - f(I, *popt)
chisq = np.sum((r / V_unc) ** 2)
print "Reduced chi squared:", chisq / ddof

fig1 = plt.figure(1)
# Plot data + model
frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
plt.scatter(I, V, label = "Data")
plt.errorbar(I, V, xerr = I_unc, yerr = V_unc, linestyle = "None")
plt.plot(I, f(I, *popt), label = "Model")
plt.title("Potential vs. Current of a Light Bulb")
plt.legend()
plt.ylabel("Potential (V)")
# frame1.set_xticklabels([])
plt.grid(True)
# Residual plot
frame2 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
plt.scatter(V, r)
plt.grid(True)
plt.xlabel("Current (mA)")
plt.ylabel("Residuals")
# plt.ylim([-2, 9])
plt.savefig("Bulb.pdf")
plt.show()
plt.close()
