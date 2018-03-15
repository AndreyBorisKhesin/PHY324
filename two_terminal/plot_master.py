# Reads .csv files (oscilloscope's) output for channels 1 and 2 and graphs the corresponding XY plot

# ---------- Import Statements ----------

import numpy as np
import matplotlib.pyplot as plt

# ---------- User-defined Functions ----------

# ---------- Main Code ----------

# Set desired font
plt.rc('font', family = 'Times New Roman')

directories = np.genfromtxt("plot_titles.txt", dtype = "str", delimiter = "\t")
print(directories)

R = 4.7	# ohms

for line in directories:
		
	ch1_file = "data\\" + line[0] + "\\F00" + line[2] + "CH1.CSV"
	ch2_file = "data\\" + line[0] + "\\F00" + line[2] + "CH2.CSV"

	# Import channel 1, channel 2 data
	ch1_data = np.genfromtxt(ch1_file, delimiter = ",")
	ch2_data = np.genfromtxt(ch2_file, delimiter = ",")

	# Extract the t, potential data only
	t = ch1_data[:, 3]
	ch1_data = ch1_data[:, 4]
	ch2_data = ch2_data[:, 4] / R

	# Determine left & right bounds of the plot (want fixed to visualize the slope)
	left_bound = np.min([np.min(ch1_data), np.min(ch2_data)])
	right_bound = np.max([np.max(ch1_data), np.max(ch2_data)])
	left_bound -= (right_bound - left_bound) / 10
	right_bound += (right_bound - left_bound) / 10

	# Graph XY graph
	plt.scatter(ch1_data, ch2_data, color = "black", s = 5)
	# plt.xlim([left_bound, right_bound])
	# plt.ylim([left_bound, right_bound])
	plt.xlabel("Device Potential (V)")
	plt.ylabel("Device Current (I)")
	plt.title(line[1])
	plt.savefig("char_curves/" + line[0] + ".pdf")
	plt.close()
