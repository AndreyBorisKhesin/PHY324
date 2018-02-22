#!/bin/python
# Linear fit of micrometer reading vs. num of fringes

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

# ---------- User-Defined Function ----------

# Linear fit function
def func(x, a, b):
	return a * x + b

# ---------- Main Code ----------

# Set desired font
plt.rc('font', family = 'Times New Roman')
path_to_file = "calibration.txt"
num, reading = np.loadtxt(path_to_file, unpack = True)
reading_unc = np.full((1, reading.size), 0.005)[0]
ddof = reading.size - 2

wavelength = 589.3e-9	# m

# Plot micrometer reading vs, fringe count
plt.plot(num, reading)
plt.grid(True)
plt.xlabel("Fringe count")
plt.ylabel("Micrometer reading")
plt.title("Micrometer reading vs. fringe count for sodium lamp")
plt.savefig("calibration1.pdf")
plt.close()

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

print "\n", "Inverse slope:", 1 / popt[0], "<-- f value used for calibration"

# Plot micrometer reading vs. carriage displacement
fig1 = plt.figure(1)
frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
plt.scatter(dist, reading, color = "black", s = 10)
plt.plot(dist, func(dist, *popt), color = "black")
plt.grid(True)
plt.ylabel("Micrometer reading")
plt.xlim([-0.3e-4, 3.3e-4])
frame1.set_xticklabels([])
plt.title("Micrometer reading vs. carriage displacement for sodium lamp")
frame2 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
plt.scatter(dist, r, color = "black", s = 10)
plt.ylabel("Residuals")
plt.xlabel("Carriage displacement (m)")
plt.xlim([-0.3e-4, 3.3e-4])
plt.grid(True)
plt.savefig("calibration2.pdf")
plt.show()
