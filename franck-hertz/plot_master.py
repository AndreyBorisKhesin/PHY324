#!/bin/python
# Plots Franck-Hertz dta & saves as a .pdf
# Assumes text files to be read are in 'data' directory, same parent directory as this .py

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt

# ---------- User-Defined Functions ----------

# ---------- Main Code ----------

# Set desired font
plt.rc('font', family = 'Times New Roman')

num_data_files = 17

# Loop over all data files, plot & save collected curves
for i in range(0, num_data_files):
	path_to_file = "data/" + str(i + 1) + ".txt"
	# path_to_file = "data\\" + str(i + 1) + ".txt"		# Windows	<===
	
	ch1_data, ch2_data = np.loadtxt(path_to_file, unpack = True)
	
	plt.plot(ch1_data, ch2_data, color = "black")
	plt.xlabel("Channel 1 Potential (V)")
	plt.ylabel("Channel 2 Potential (V)")
	plt.title("Current vs. accelerating voltage")
	plt.savefig(str(i + 1) + ".pdf")
	plt.close()

