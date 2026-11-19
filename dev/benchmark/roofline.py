#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


# Define NVIDIA A100 memory bandwidth (in Bytes/s)
memory_bandwidth = {
  "HBM2": 1555e9
  # ,  # HBM2 memory (typical)
  # "DRAM": (2 * 8 * 17.6e9),  # Assuming 2 channels, 8 ranks, and 17.6 GB/s per rank
}

# Define peak performance (in FLOP/s) for A100 (approximate)
peak_flops = 9.7e12  # TensorFloat-32 (TF32) peak performance (approximate)

# Create x-axis (arithmetic intensity)
ai_min = 1
ai_max = 100
arithmetic_intensity = np.array([x for x in range(int(ai_min * 100), int(ai_max * 100) + 1)])/ 100
# print(arithmetic_intensity)
def compute_roofline(bandwidth, flops):
	return np.minimum(bandwidth * arithmetic_intensity, flops)

# Plot rooflines
for level, bandwidth in memory_bandwidth.items():
	gflops = compute_roofline(bandwidth, peak_flops)
	# print(gflops)
	gflops = gflops/1e9
	plt.plot(arithmetic_intensity, gflops, label=f"{level} ({bandwidth/1e12} TB/s) ")

	idx = 0
	for i in range(0, len(gflops) -1 ):
		if not gflops[i+1] > gflops[i]:
			idx = i
			break
	
	print(idx, arithmetic_intensity[idx], gflops[idx])
	# plt.text(arithmetic_intensity[idx], gflops[idx], f"{gflops[idx]} GFLOP/s", ha="center", va="center")

	# arrow_props = dict(facecolor='black', shrink=0.05)  # Customize arrow properties
	plt.annotate(
		f"{round(gflops[idx]/1000, 1)} TFLOP/s", 
		xy=(arithmetic_intensity[idx], gflops[idx]), 
		xytext=(arithmetic_intensity[idx], gflops[idx] + 1),
		verticalalignment='bottom',weight="bold")


tet10_AI = 19.16
tet10_gflops = 5 * 1000
plt.plot(tet10_AI, tet10_gflops, label=f"tet10 ", marker = 'o', linestyle='None')

plt.annotate(
	f"5 TFLOP/s", 
	xy=(tet10_AI, tet10_gflops), 
	xytext=(tet10_AI+1, tet10_gflops),
	horizontalalignment='left',
	verticalalignment='bottom',weight="bold")

# Plot peak performance line
# plt.plot(arithmetic_intensity, [peak_flops] * len(arithmetic_intensity), '--', label="Peak Performance (TF32)")

# Configure plot
plt.xlabel("Arithmetic Intensity (FLOP/Byte)")
plt.ylabel("Performance (GFLOP/s)")
plt.title("Naive Roofline Model - NVIDIA A100 (HBM2)")
plt.grid(True)
plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.grid(True)  # Optional: Add grid for better readability
plt.grid(True, which='minor', linestyle='--')
# plt.tight_layout()
# plt.show()
plt.savefig('roofline.pdf')
