#!/usr/bin/python
# Fits data for helium to the Hartmann relation method

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------- User-Defined Functions ----------

# Test function (Hartmann relation, m = a, b = b)
# (x, a, b) --> (f)
# (float, float, float) --> (float)
def f(x, a, b):
	return a / (x - lambda_0) + b

# ---------- Main Code ----------

unc = 0.02
lambda_0 = 285.2	# (+- 0.4) nm

# Read .txt file & extract data
path_to_file = "/home/polina/Documents/3rd_Year/PHY324/Atomic Spectra/atomic-spectra-helium.txt"
scale, wavelength = np.loadtxt(path_to_file, unpack = True, usecols = (0, 1))
# Array with uncertainties
scale_unc = np.empty(scale.size)
scale_unc.fill(unc)
# Degrees of freedom
ddof = np.size(wavelength - 2)

# Fit data
popt, pcov = curve_fit(f, wavelength, scale, sigma = scale_unc)
# Calculate chi squared + residuals
r = scale - f(wavelength, *popt)
chisq = np.sum(r ** 2 / scale_unc)
print "Reduced chi squared:", chisq / ddof

print "Parameters:", popt

# Generate data for plotting (using fit parameters)
x_fit = np.linspace(np.amin(wavelength), np.amax(wavelength), 100)
y_fit = f(x_fit, *popt)

# Plot data + model with fit parameters
plt.plot(wavelength, scale, label = "Experiment")
plt.plot(x_fit, y_fit, label = "Fit")
plt.legend()
plt.grid(True)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Scale reading")
plt.title("Scale reading vs. Wavelength for Helium Gas")
plt.savefig("Hartmann-fit.pdf")
plt.close()

plt.scatter(wavelength,r)
plt.plot(np.linspace(np.amin(wavelength) - 100, np.amax(wavelength) + 100, np.size(wavelength)), np.zeros(np.size(wavelength)))
plt.xlabel("Wavelength (nm)")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.xlim([400, 800])
plt.ylim([-0.15, 0.15])
plt.grid()
plt.savefig("Residuals.pdf")
plt.close()
