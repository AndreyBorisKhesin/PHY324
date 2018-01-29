#!/bin/python
# Filters data to get the maxima curve
# Go back & fit using np.polyfit

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

# ---------- User-Defined Function ----------

# ---------- Main Code ----------

# Set desired font
plt.rc('font', family = 'Times New Roman')
path_to_file = "/home/polina/Documents/3rd_Year/PHY324/polarization/exercise3-no-polarizer.txt"
position_raw, intensity_raw = np.loadtxt(path_to_file, unpack = True)

# Sort the two original arrays
permutation = position_raw.argsort()
position_raw = position_raw[permutation]
intensity_raw = intensity_raw[permutation]

position = np.array([])
intensity = np.array([])

# Number of ipoints per interval for selecting max
pts_per_interval = 300
for i in range(0, position_raw.size, pts_per_interval):
	# Check not working w/ last interval:
	if i < position_raw.size - pts_per_interval:
		temp_index = np.argmax(intensity_raw[i:i + pts_per_interval])
		position = np.append(position, position_raw[i:i + pts_per_interval][temp_index])
		intensity = np.append(intensity, intensity_raw[i:i + pts_per_interval][temp_index])
	# Last interval
	else:
		temp_index = np.argmax(intensity_raw[i:])
		position = np.append(position, position_raw[i:][temp_index])
		intensity = np.append(intensity, intensity_raw[i:][temp_index])

# Fit filtered data to a polynomial
coeffs =  np.polyfit(position, intensity, 7, full = False)
polynomial_fit = np.poly1d(coeffs)
intensity_fit = polynomial_fit(position)

plt.scatter(position_raw, intensity_raw, s = 5, color = "black")
plt.xlabel("Sensor Position (degrees)")
plt.ylabel("Light Intentisty (V)")
plt.ylim([0, 1.2])
plt.title("Intensity vs. position, polarization by refraction")
plt.grid()
plt.savefig("exercise3-data.pdf")
plt.show()
plt.close()

# Plot original (collected data)
plt.subplot(121)
plt.scatter(position_raw, intensity_raw, s = 5, color = "black")
plt.plot(position, intensity_fit, color = "red")
plt.xlabel("Sensor Position (degrees)")
plt.ylabel("Light Intentisty (V)")
plt.ylim([0, 1.2])
plt.gca().set_title('Collected data')
# Plot filtered data
plt.subplot(122)
plt.scatter(position, intensity, s = 5, color = "black")
# plt.plot(position, intensity_fit, color = "red")
plt.xlabel("Sensor Position (degrees)")
plt.ylabel("Light Intensity (V)")
plt.ylim([0, 1.2])
plt.gca().set_title('Filtered data')
plt.show()

