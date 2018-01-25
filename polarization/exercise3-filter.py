#!/bin/python
# Filters data to get the maxima curve
# Go back & fit using np.polyfit

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------- User-Defined Function ----------

# Fit function (quartic)
def func(x, a, b, c, d, e): 
	return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

# ---------- Main Code ----------

# Set desired font
plt.rc('font', family = 'Times New Roman')
path_to_file = "/home/polina/Documents/3rd_Year/PHY324/polarization/exercise3-no-polarizer.txt"
position_raw, intensity_raw = np.loadtxt(path_to_file, unpack = True)

# Filtered arrays
position = np.array([])
intensity = np.array([])
for i in range(0, position_raw.size):
	# First data point for this position
	if position_raw[i] not in position:
		position = np.append(position, position_raw[i])
		intensity = np.append(intensity, intensity_raw[i])
	# Position aleready in array
	else:
		if intensity_raw[i] >= intensity[ np.where(position == position_raw[i])[0][0] ]:
			intensity[ np.where(position == position_raw[i])[0][0] ] = intensity_raw[i]

# Uncertainty in intensity
I_unc = 0.01
intensity_unc = np.full(intensity.size, I_unc)

# Number of fit parameters
num_parameters = 5
# Find degrees of freedom for redeced chi squared
ddof = intensity.size - num_parameters

print np.sort(position)

# Fit data
popt, pcov = curve_fit(func, position, intensity, sigma = intensity_unc)
# Find residuals
r = intensity - func(position, *popt)
chisq = np.sum((r / intensity_unc) ** 2)
print "Reduced chi squared:", chisq / ddof
# Calculate and print R^2
ss_res = np.sum(r ** 2)
ss_tot = np.sum((intensity - np.mean(intensity)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print "R^2:", r_squared

fig1 = plt.figure(1)
# Add raw data (all collected intensitites)
frame1 = fig1.add_subplot(121)
plt.scatter(position_raw, intensity_raw, s = 5, color = "black")
plt.title("All data points")
# Add filtered data (maximum intensitites in each position array)
frame2 = fig1.add_subplot(122)
plt.scatter(position, intensity, s = 5, color = "black")
plt.plot(np.sort(position), func(np.sort(position), *popt))
plt.title("Filtered data")
plt.show()
