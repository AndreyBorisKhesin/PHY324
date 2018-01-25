#!/usr/bin/python
# Creates a single data file for the three polarizer data (exercise 2)

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt

# ---------- User-Defined Functions ----------

# ---------- Main Code ----------

path_to_file_1 = "/home/polina/Documents/3rd_Year/PHY324/polarization/exercise2-1.txt"
path_to_file_2 = "/home/polina/Documents/3rd_Year/PHY324/polarization/exercise2-2.txt"
pos_1, I_1 = np.loadtxt(path_to_file_1,unpack = True)
pos_2, I_2 = np.loadtxt(path_to_file_2,unpack = True)

# Shift second  
pos_2 = pos_2 + np.amax(pos_1)

position = np.concatenate((pos_1, pos_2))
intensity = np.concatenate((I_1, I_2))

plt.plot(position, intensity)
plt.show()

