#!/usr/bin/python
# Plots and fits data for a thermistor w/ varying temperatures

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------- User-Defined Functions ----------

# Fit function for optimizing; 
def f(x, a, b):
	return a * np.exp(- x / b)

# ---------- Main Code ----------

path_to_file = "/home/polina/Documents/3rd_Year/PHY324/circuit-elements/thermistor.txt"
temp, resist = np.loadtxt(path_to_file, unpack = True)
temp = temp + 273.0			# Convert temperature to K

print(temp)
print(resist)

# Arrays w/ uncertainties
temp_unc = np.full(np.size(temp), 0.5)
resist_unc = np.concatenate((np.full(12, 0.01), np.full(18, 0.1)))

popt, pcov = curve_fit(f, temp, resist)

plt.plot(temp, resist)
plt.grid(True)
plt.xlabel("Temperature (C)")
plt.ylabel("Resistance (ohms)")
plt.title("Resistance vs. Temperature for a Thermistor")
plt.show()
