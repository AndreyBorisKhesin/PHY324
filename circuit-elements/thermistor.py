#!/usr/bin/python
# Plots and fits data for a thermistor w/ varying temperatures

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt

# ---------- User-Defined Functions ----------

def f(x, a, b):
	

# ---------- Main Code ----------

path_to_file = "/home/polina/Documents/3rd_Year/PHY324/circuit-elements/thermistor.txt"
temp, resist = np.loadtxt(path_to_file, unpack = True)



plt.plot(temp, resist)
plt.grid(True)
plt.xlabel("Temperature (C)")
plt.ylabel("Resistance (ohms)")
plt.title("Resistance vs. Temperature for a Thermistor")
plt.show()
