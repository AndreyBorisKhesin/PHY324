#!/bin/python
# Plots temperature change and electrical power input versus time

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

# ---------- User-Defined Function ----------

def func(x, a, b, c):
	return a + b * np.exp(c * x)

# ---------- Main Code ----------

plt.rc('font', family = 'Times New Roman')
# path_to_file = "/home/polina/Documents/3rd_Year/PHY324/thermoelectricity/part2a.txt"
path_to_file = "part2a.txt"
V, I, T1, T2 = np.loadtxt(path_to_file, unpack = True)

P = V * I
delta_T = T1 - T2
delta_T_unc = np.full((1, delta_T.size), np.sqrt(2) * 0.1)[0]

ddof = delta_T.size - 3

# Time array
t = np.arange(0, 1.0 * delta_T.size, 1.0)

popt, pcov = curve_fit(func, t, delta_T, p0 = [37, -37, -1/14])
print "a:", popt[0], "+-", np.sqrt(pcov[0, 0])
print "b:", popt[1], "+-", np.sqrt(pcov[1, 1])
print "c:", popt[2], "+-", np.sqrt(pcov[2, 2])
# Find residuals
r = delta_T - func(t, *popt)
chisq = np.sum((r / delta_T_unc) ** 2)
print "Reduced chi squared:", chisq / ddof

# Calculate and print R^2
ss_res = np.sum(r ** 2)
ss_tot = np.sum((delta_T - np.mean(delta_T)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print "R^2:", r_squared

figure = plt.figure(num = None, figsize = (10, 6), dpi = 80, facecolor = 'w')
# Plot power
ax1 = figure.add_subplot(221)
plt.plot(P, color = "red", label = "Power (W)")
plt.grid(True)
plt.gca().set_title("Power input")
plt.xticks([], [])
plt.ylabel("Power (W)")
# Plot temperature difference
ax2 = figure.add_subplot(222)
plt.plot(t, delta_T, color = "blue", label = "Temperature difference (K)")
plt.plot(t, func(t, *popt))
plt.gca().set_title("Temperature change")
plt.xticks([], [])
plt.grid(True)
plt.ylabel("Temperature difference (K)")
plt.suptitle("TEC data, Varniac autotransformer turned off")
ax3 = figure.add_subplot(313)
plt.scatter(t, r)
plt.grid(True)
plt.xlabel("Time")
plt.xticks([], [])
plt.ylabel("Residuals")
ax3.title.set_text("Temperature difference vs. time fit residuals")
plt.savefig("part2a-data.pdf")
plt.show()

