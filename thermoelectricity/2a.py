#!/bin/python
# Plots temperature change and electrical power input versus time

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

# ---------- User-Defined Function ----------

# ---------- Main Code ----------

plt.rc('font', family = 'Times New Roman')
# path_to_file = "/home/polina/Documents/3rd_Year/PHY324/thermoelectricity/part2a.txt"
path_to_file = "part2a.txt"
V, I, T1, T2 = np.loadtxt(path_to_file, unpack = True)

P = V * I
delta_T = T1 - T2

plt.figure(num = None, figsize = (10, 6), dpi = 80, facecolor = 'w')
# Plot power
plt.subplot(121)
plt.plot(P, color = "red", label = "Power (W)")
plt.grid(True)
plt.gca().set_title("Power input")
plt.xticks([], [])
plt.ylabel("Power (W)")
# Plot temperature difference
plt.subplot(122)
plt.plot(delta_T, color = "blue", label = "Temperature difference (K)")
plt.gca().set_title("Temperature change")
plt.xticks([], [])
plt.grid(True)
plt.ylabel("Temperature difference (K)")
plt.suptitle("TEC data, Varniac autotransformer turned off")
plt.savefig("part2a-data.pdf")
plt.show()

