#!/bin/python
# Filters data to get the maxima curve
# Go back & relabel angles --> software took them incorrectly

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

# ---------- User-Defined Function ----------

# ---------- Main Code ----------

# Set desired font
plt.rc('font', family = 'Times New Roman')
path_to_file = "C:\\Users\\Andrey\\Documents\\PHY324\\polarization\\exercise3-no-polarizer.txt"
position_raw, intensity_raw = np.loadtxt(path_to_file, unpack = True)

# Sort the two original arrays
permutation = position_raw.argsort()
position_raw = position_raw[permutation]
intensity_raw = intensity_raw[permutation]
position_raw = 120.0 - np.abs(180.0 - position_raw)

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

position = position[1:]
intensity = intensity[1:]
# Fit filtered data to a polynomial
coeffs =  np.polyfit(position, intensity, 3, full = False)
polynomial_fit = np.poly1d(coeffs)
intensity_fit = polynomial_fit(position)

# Find the local minima
crit = polynomial_fit.deriv().r
r_crit = crit[crit.imag==0].real
test = polynomial_fit.deriv(2)(r_crit) 
# Find local minima (excluding endpoints)
x_min = r_crit[test > 0]
y_min = polynomial_fit(x_min)
print("Coefficients:", coeffs)
# Find residuals
r = intensity - intensity_fit
chisq = np.sum((r / 0.01) ** 2)
print("Reduced chi squared:", chisq / (intensity.size - 4))
# Calculate and print R^2
ss_res = np.sum(r ** 2)
ss_tot = np.sum((intensity - np.mean(intensity)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print("R^2:", r_squared)
# Brewster's angle
print("Minimum angle (Brewster angle):", x_min[0] / 2)
n1 = 1.00 	# Index of refraction of intial medium (air)
# Determine refractive index of acrylic
p_angle = x_min[0] / 2 * np.pi / 180.0
n2 = n1 * np.tan(p_angle)
print("Refractive index of acrylic:", n2)

# Determine reflection coefficients & reflectances
theta_t = np.arcsin(n1 * np.sin(p_angle) / n2)		# Refracted angle
r_perp = (n1 * np.cos(p_angle) - n2 * np.cos(theta_t)) / (n1 * np.cos(p_angle) + n2 * np.cos(theta_t))
r_parallel = (n1 * np.cos(theta_t) - n2 * np.cos(p_angle)) / (n1 * np.cos(theta_t) + n2 * np.cos(p_angle))
print "Normal reflectance:", r_perp ** 2
print "Parallel reflectance:", r_parallel ** 2

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
plt.xlim([94, 121])
plt.ylim([0, 1.2])
plt.gca().set_title('Collected data')
# Plot filtered data
plt.subplot(122)
plt.scatter(position, intensity, s = 5, color = "black")
# plt.plot(x_min, y_min, "o")
# plt.plot(position, intensity_fit, color = "red")
plt.xlabel("Sensor Position (degrees)")
plt.ylabel("Light Intensity (V)")
plt.xlim([94, 121])
plt.ylim([0, 1.2])
plt.gca().set_title('Filtered data')
plt.savefig("exercise3-model.pdf")
plt.close()

position_hor, intensity_hor = np.loadtxt("/home/polina/Documents/3rd_Year/PHY324/polarization/exercise3-horizontal.txt", unpack = True)
position_vert, intensity_vert = np.loadtxt("/home/polina/Documents/3rd_Year/PHY324/polarization/exercise3-vertical.txt", unpack = True)

position_hor = 120.0 - np.abs(180.0 - position_hor)
position_vert = 120.0 - np.abs(180.0 - position_vert)

# Plot horizontal polarizer intensity
plt.scatter(position_hor, intensity_hor, s = 5, color = "black")
plt.xlabel("Sensor Position (degrees)")
plt.ylabel("Light Intentisty (V)")
plt.title("Intensity vs. angluar position for horizontal polarizer")
plt.grid(True)
plt.savefig("exercise3-horizontal.pdf")
plt.show()
plt.close()

# Plot horizontal polarizer intensity
plt.scatter(position_vert, intensity_vert, s = 5, color = "black")
plt.xlabel("Sensor Position (degrees)")
plt.ylabel("Light Intentisty (V)")
plt.title("Intensity vs. angluar position for vertical polarizer")
plt.grid(True)
plt.savefig("exercise3-vertical.pdf")
plt.show()
plt.close()
