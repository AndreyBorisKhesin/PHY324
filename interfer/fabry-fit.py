#!/bin/python
# Calibration curve for the Fabry interferometer

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

# ---------- User-Defined Function ----------

# Linear fit function
def func(x, a, b):
	return a * x + b
	# return a * x
	
# ---------- Main Code ----------

# Set desired font
plt.rc('font', family = 'Times New Roman')
path_to_file = "fabry-measurements.txt"
num, reading = np.loadtxt(path_to_file, unpack = True)
reading_unc = np.full((1, reading.size), 0.005)[0]
num = np.flip(num, 0)
ddof = reading.size - 2

wavelength = 589.3e-9	# m

dist = num * wavelength / 2		# m

popt, pcov = curve_fit(func, dist, reading, sigma = reading_unc)
print "a:", popt[0], "+-", np.sqrt(pcov[0, 0]), "(slope)"
print "b:", popt[1], "+-", np.sqrt(pcov[1, 1])
# Calculate chi squared + residuals
r = reading - func(dist, *popt)
chisq = np.sum((r / reading_unc) ** 2)
print "Reduced chi squared:", chisq / ddof
# Calculate and print R^2
ss_res = np.sum(r ** 2)
ss_tot = np.sum((reading - np.mean(reading)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print "R^2:", r_squared

# Plot micrometer reading vs. fringe count
fig1 = plt.figure(1)
frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
plt.scatter(dist, reading, color = "black", s = 5)
plt.plot(dist, func(dist, *popt), color = "black")
plt.grid(True)
plt.ylabel("Micrometer reading")
plt.title("Micrometer reading vs. carriage displacement for sodium lamp, Fabry-Perot")
plt.xlim([-0.00005, 0.00035])
frame1.set_xticklabels([])
# Plot residuals
frame2 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
plt.scatter(dist, r, s = 5, color = "black")
plt.xlabel("Carriage displacement (m)")
plt.ylabel("Residuals")
plt.xlim([-0.00005, 0.00035])

plt.grid(True)
plt.savefig("calibration3.pdf")
plt.show()
plt.close()
