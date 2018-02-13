#!/bin/python
# Plots temperature change and electrical power input versus time

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

# ---------- User-Defined Function ----------

def func(x, a, b):
	return a * x + b

# ---------- Main Code ----------

plt.rc('font', family = 'Times New Roman')
path_to_file = "part1.txt"
T1, T1_unc, T2, T2_unc, I, I_unc, V, V_unc = np.loadtxt(path_to_file, unpack = True)

T0 = 22.1
T0_unc = 0.1

P = V * I
P_unc = np.abs(P) * np.sqrt((I_unc / I) ** 2 + (V_unc / V) ** 2)
T_unc = np.sqrt(2 * T1_unc ** 2)

ddof = T1.size - 2

print "T_in - T_out"
popt1, pcov1 = curve_fit(func, P, T2 - T1)
print "a:", popt1[0], "+-", np.sqrt(pcov1[0, 0])
print "b:", popt1[1], "+-", np.sqrt(pcov1[1, 1])
# Find residuals
r1 = (T2 - T1) - func(P, *popt1)
chisq = np.sum((r1 / T_unc) ** 2)
print "Reduced chi squared:", chisq / ddof
# Calculate and print R^2
ss_res = np.sum(r1 ** 2)
ss_tot = np.sum((T2 - T1 - np.mean(T2 - T1)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print "R^2:", r_squared

print "\n K_d", 1 / popt1[0], "+-", np.sqrt(pcov1[0, 0]) / (popt1[0] ** 2), "\n"

print "T_out - T_0"
popt2, pcov2 = curve_fit(func, P, T1 - T0)
print "a:", popt2[0], "+-", np.sqrt(pcov2[0, 0])
print "b:", popt2[1], "+-", np.sqrt(pcov2[1, 1])

# Find residuals
r2 = (T1 - T0) - func(P, *popt2)
chisq = np.sum((r2 / T_unc) ** 2)
print "Reduced chi squared:", chisq / ddof
# Calculate and print R^2
ss_res = np.sum(r2 ** 2)
ss_tot = np.sum((T1 - T0 - np.mean(T1 - T0)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print "R^2:", r_squared

print "\n K_hs", 1 / popt2[0], "+-", np.sqrt(pcov2[0, 0]) / (popt2[0] ** 2), "\n"

fig1 = plt.figure(1)
frame1 = fig1.add_axes((0.1, 0.3, 0.8, 0.6))
plt.plot(P, T2 - T1, label = "T_in - T_out", color = "red")
plt.errorbar(P, T2 - T1, xerr = P_unc, yerr = T_unc, linestyle = "None", color = "black")
plt.plot(P, func(P, *popt1))
plt.plot(P, T1 - T0, label = "T_out - T_0", color = "green")
plt.errorbar(P, T1 - T0, xerr = P_unc, yerr = T_unc, linestyle = "None", color = "black")
plt.plot(P, func(P, *popt2))
plt.grid()
plt.title("Temperature difference vs. input power")
plt.ylabel("Temperature (Degrees)")
plt.xlim([-0.2, 5.2])
plt.legend()
frame2 = fig1.add_axes((0.1, 0.1, 0.8, 0.2))
plt.scatter(P, r1, color = "red")
plt.scatter(P, r2, color = "green")
plt.xlabel("Power (W)")
plt.ylabel("Residuals")
plt.xlim([-0.2, 5.2])
plt.grid(True)
plt.savefig("1.pdf")
plt.show()
