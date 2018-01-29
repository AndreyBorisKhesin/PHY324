#!/usr/bin/python
# Fits obtained data for two polarizers for Intensity vs. cos(theta) and
# Intensity vs. (cos(theta)) ** 2

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------- User-Defined Functions ----------

# Fit function for optimizing #1
# Intensity vs. cos(theta), a = I_0, initial intensity
# (x, a) --> (f_value)
# (float, float) --> (float)
def f1(x, a):
	return a * np.cos(x)

# Fit function for optimizing #2
# Intensity vs. (cos(theta)) ** 2, a = I_0, initial intensity
# (x, a) --> (f_value)
# (float, float) --> (float)
def f2(x, a):
	return a * np.cos(x) ** 2

# ---------- Main Code ----------

# Uncertainty in intensity
I_unc = 0.01

# Set desired font
plt.rc('font', family = 'Times New Roman')
path_to_file = "/home/polina/Documents/3rd_Year/PHY324/polarization/exercise1-1.txt"
position, intensity = np.loadtxt(path_to_file, unpack = True)
intensity_unc = np.full(intensity.size, I_unc)

# Plot raw data
plt.plot(position, intensity)
plt.grid(True)
plt.xlabel("Sensor Position (rad)")
plt.ylabel("Light Intensity (V)")
plt.title("Intensity vs. Position for Two Polarizers")
plt.savefig("exercise1-data.pdf")
plt.close()

# Number of fit parameters
num_parameters = 1
# Find degrees of freedom for redeced chi squared
ddof = intensity.size - num_parameters

# Fit intensity vs. (cos(theta)) fitting
print "Fitting to cos(theta)"
popt, pcov = curve_fit(f1, position, intensity, sigma = intensity_unc)
print "I_0:", popt[0], "+-", np.sqrt(pcov[0, 0])
# Find residuals
r = intensity - f1(position, *popt)
chisq = np.sum((r / intensity_unc) ** 2)
print "Reduced chi squared:", chisq / ddof
# Calculate and print R^2
ss_res = np.sum(r ** 2)
ss_tot = np.sum((intensity - np.mean(intensity)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print "R^2:", r_squared

# Fit intensity vs. (cos(theta)) ** 2 fitting
print "Fitting to cos(theta) ** 2"
popt, pcov = curve_fit(f2, position, intensity, sigma = intensity_unc)
print "I_0:", popt[0], "+-", np.sqrt(pcov[0, 0])
# Find residuals
r = intensity - f2(position, *popt)
chisq = np.sum((r / intensity_unc) ** 2)
print "Reduced chi squared:", chisq / ddof
# Calculate and print R^2
ss_res = np.sum(r ** 2)
ss_tot = np.sum((intensity - np.mean(intensity)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print "R^2:", r_squared

# Plot and save raw data only
plt.scatter(position, intensity, label = "Data", s = 10, color = "black")
plt.errorbar(position, intensity, yerr = intensity_unc, linestyle = "None", color = "black")
plt.title("Intensity vs. position for two polarizers")
plt.xlabel("Sensor Position (rad)")
plt.ylabel("Light Intensity (V)")
plt.grid(True)
plt.title("Intensity vs. position data for two polarizers")
plt.savefig("exercise1-data.pdf")
# plt.show()
plt.close()

fig1 = plt.figure(1)
# Plot data + model
frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
plt.scatter(position, intensity, label = "Data", s = 5, color = "black")
plt.errorbar(position, intensity, yerr = intensity_unc, linestyle = "None", color = "black")
plt.plot(position, f2(position, *popt), label = "Model", color = "red")
plt.title("Intensity vs. position for two polarizers")
plt.ylabel("Light Intensity (V)")
#plt.xlim([-0.2, 3.2])
frame1.set_xticklabels([])
plt.grid(True)
# Residual plot
frame2 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
plt.scatter(position, r, s = 10, color = "black")
plt.grid(True)
plt.xlabel("Sensor Position (rad)")
plt.ylabel("Residuals")
#plt.xlim([-0.2, 3.2])
#plt.ylim([-3, 9])
plt.savefig("exercise1-model.pdf")
plt.show()
plt.close()
