#!/usr/bin/python
# Fits obtained data for three polarizers

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------- User-Defined Functions ----------

# Function for fitting:

def f(x, a):
	return (a / 4) * np.sin(2 * (x)) ** 2
	# return (a / 4) * np.sin(2 * (x + np.pi / 4)) ** 2

# ---------- Main Code ----------

# Set desired font
plt.rc('font', family = 'Times New Roman')
# Concatenate data files to conver 360 degrees
path_to_file_1 = "C:\\Users\\Andrey\\Documents\\PHY324\\polarization\\exercise2-1.txt"
path_to_file_2 = "C:\\Users\\Andrey\\Documents\\PHY324\\polarization\\exercise2-2.txt"
pos_1, I_1 = np.loadtxt(path_to_file_1,unpack = True)
pos_2, I_2 = np.loadtxt(path_to_file_2,unpack = True)
# Shift second  
pos_2 = pos_2 + np.amax(pos_1)
# Final data arrays
position = np.concatenate((pos_1, pos_2))
intensity = np.concatenate((I_1, I_2))

# Uncertainty in intensity
I_unc = 0.01
intensity_unc = np.full(intensity.size, I_unc)

# Number of fit parameters
num_parameters = 1
# Find degrees of freedom for redeced chi squared
ddof = intensity.size - num_parameters

# Fit intensity vs. (cos(theta)) fitting
print("Fitting to sin(2(theta)) ** 2")
popt, pcov = curve_fit(f, position, intensity, sigma = intensity_unc)
print("I1:", popt[0], "+-", np.sqrt(pcov[0, 0]))
# Find residuals
r = intensity - f(position, *popt)
chisq = np.sum((r / intensity_unc) ** 2)
print("Reduced chi squared:", chisq / ddof)
# Calculate and print R^2
ss_res = np.sum(r ** 2)
ss_tot = np.sum((intensity - np.mean(intensity)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print("R^2:", r_squared)

fig1 = plt.figure(1)
# Plot data + model
frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
plt.scatter(position, intensity, label = "Data", s = 10, color = "black")
plt.title("Intensity vs. position for three polarizers")
plt.ylabel("Light Intensity (V)")
plt.plot(position, f(position, *popt), color = "red")
frame1.set_xticklabels([])
plt.grid(True)
# Residual plot
frame2 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
plt.scatter(position, r, s = 10, color = "black")
plt.grid(True)
plt.xlabel("Sensor Position (rad)")
plt.ylabel("Residuals")
plt.savefig("exercise2-model.pdf")
plt.show()
plt.close()

