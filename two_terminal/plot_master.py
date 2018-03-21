#!/bin/python
# Reads .csv files (oscilloscope's) output for channels 1 and 2 and graphs the corresponding XY plot

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------- User-defined Functions ----------

# Linear model function used for fitting
def func(x, a, b):
	return a * x + b

# ---------- Main Code ----------

# Set desired font
plt.rc('font', family = 'Times New Roman')

directories = np.genfromtxt("plot_titles.txt", dtype = "str", delimiter = "\t")

R = 4.7	# ohms

for line in directories:
		
	ch1_file = "data/" + line[0] + "/F00" + line[2] + "CH1.CSV"
	ch2_file = "data/" + line[0] + "/F00" + line[2] + "CH2.CSV"
		
	# ch1_file = "data\\" + line[0] + "\\F00" + line[2] + "CH1.CSV"
	# ch2_file = "data\\" + line[0] + "\\F00" + line[2] + "CH2.CSV"

	# Import channel 1, channel 2 data
	ch1_data = np.genfromtxt(ch1_file, delimiter = ",")
	ch2_data = np.genfromtxt(ch2_file, delimiter = ",")

	# Extract the t, potential data only
	t = ch1_data[:, 3]
	ch1_data = ch1_data[:, 4]
	ch2_data = (ch2_data[:, 4] / R)

	# Index of elements whose i vs. v is linear (can fit for slope)
	need_slope = np.array(["1", "2", "3", "4a", "4b", "10"])

	print(line)
	'''
	unc_ch1 = ch1_data * 0.04
	unc_ch2 = ch2_data * 0.04
	ddof = 2
	# Print slope of the relation (if linear)
	if line[0][ : (line[0]).find("-") ] in need_slope:
		popt, pcov = curve_fit(func, ch1_data, ch2_data, sigma = unc_ch2)
		print "Slope:", popt[0], "+-", np.sqrt(pcov[0, 0])
		# Find residuals
		r1 = ch2_data - func(ch1_data, *popt)
		chisq = np.sum((r1 / unc_ch2) ** 2)
		print "Reduced chi squared:", chisq / ddof
		# Calculate and print R^2
		ss_res = np.sum(r1 ** 2)
		ss_tot = np.sum((ch2_data - np.mean(ch2_data)) ** 2)
		r_squared = 1 - (ss_res / ss_tot)
		print "R^2:", r_squared
	'''

	# Determine left & right bounds of the plot (want fixed to visualize the slope)
	left_bound = np.min([np.min(ch1_data), np.min(ch2_data)])
	right_bound = np.max([np.max(ch1_data), np.max(ch2_data)])
	left_bound -= (right_bound - left_bound) / 10
	right_bound += (right_bound - left_bound) / 10

	# Graph XY graph
	# plt.scatter(ch1_data, ch2_data, color = "black", s = 10)
	plt.plot(ch1_data, ch2_data, color = "black")
	# plt.xlim([left_bound, right_bound])
	# plt.ylim([left_bound, right_bound])
	plt.xlabel("Device Potential (V)")
	plt.ylabel("Device Current (I)")
	plt.title(line[1])
	plt.savefig("char_curves/" + line[0] + ".pdf")
	if line[0][ : (line[0]).find("-") ] in need_slope:
		plt.show()
	plt.close()
