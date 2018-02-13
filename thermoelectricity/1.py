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
path_to_file = "part1.txt"
T1, T1_unc, T2, T2_unc, I, I_unc, V, V_unc = np.loadtxt(path_to_file, unpack = True)

T0 = 22.1
T0_unc = 0.1

P = V * I
P_unc = np.abs(P) * np.sqrt((I_unc / I) ** 2 + (V_unc / V) ** 2)
T_unc = np.sqrt(2 * T1_unc ** 2)
print(P_unc)

plt.plot(P, T2 - T1, label = "T_in - T_out", color = "red")
plt.errorbar(P, T2 - T1, xerr = P_unc, yerr = T_unc, linestyle = "None", color = "black")
plt.plot(P, T1 - T0, label = "T_out - T_0", color = "green")
plt.errorbar(P, T1 - T0, xerr = P_unc, yerr = T_unc, linestyle = "None", color = "black")
plt.grid()
plt.title("Temperature difference vs. input power")
plt.xlabel("Power (W)")
plt.ylabel("Temperature (Degrees)")
plt.legend()
plt.savefig("1.pdf")
plt.show()
