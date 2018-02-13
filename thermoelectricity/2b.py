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
# path_to_file = "/home/polina/Documents/3rd_Year/PHY324/thermoelectricity/part2b.txt"
path_to_file = "part2b.txt"
V_s, V_d, I_d, T1, T2, V_in, I_in = np.loadtxt(path_to_file, unpack = True)

P_in = V_in * I_in
delta_T = T1 - T2
P_d = V_d * I_d

K_d = 0.58858351619
K_d_unc = 0.0582914163177 

# S_d = (V_in * I_in + 0.5 * V_d * I_d + K_d * (T1 - T2)) / (T2 * I_d)
S_d = V_s / (T1 - T2)
print "S_d:", np.mean(S_d), "+-", np.std(S_d)

R_d = V_d / I_d
print "R_d:", np.mean(R_d), "+-", np.std(R_d)

# Plot data
plt. figure(num = None, figsize = (14, 8), dpi = 80, facecolor = 'w')
# Temperature difference
plt.subplot(131)
plt.plot(P_in, delta_T)
plt.grid(True)
plt.xlabel("P_in (W)")
plt.ylabel("Temperature difference (K)")
plt.title("Temperature difference vs. input power")
# P_d
plt.subplot(132)
plt.plot(P_in, P_d)
plt.grid(True)
plt.xlabel("P_in (W)")
plt.ylabel("P_d (W)")
plt.title("Total electrical power vs. input power")
# V_d
plt.subplot(133)
plt.plot(P_in, V_d)
plt.grid(True)
plt.xlabel("P_in (W)")
plt.ylabel("V_d (V)")
plt.title("Total voltage across TEC vs. input power")
# Show plots
plt.suptitle("TEC data, Varniac autotransformer turned on")
plt.savefig("part2b-data.pdf")
plt.show()
