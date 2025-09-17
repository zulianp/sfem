using DelimitedFiles
using Plots
using Printf

# Set the backend to gr, which is generally more reliable
gr()

function plot_bench(csv_file::String, output_file::String)
	data = readdlm(csv_file, ',', skipstart = 1)

	println("Data read from $csv_file:")
	println(data)

	# Assuming data has at least two columns: x (e.g., size or index) and y (e.g., time or metric)
	# If more columns, you can plot multiple series
	if size(data, 2) >= 2
		x = data[:, 1]  # First column as x-axis
		y = data[:, 4]  # Second column as y-axis

		# Create the plot
		p = plot(x, y, label = "GH200 fp32 adjoint tet4",
			xlabel = "Cluster size", ylabel = "Throughput (elements/sec)",
			title = "", xaxis = :log,
			yformatter = x -> @sprintf("%.1e", x),
			xformatter = x -> @sprintf("%d", x),
			xminorticks = true,
			xminorgrid = true,
			yminorgrid = true,
			linewidth = 2.5)


		display(p)

		# Save the plot
		savefig(p, output_file)
		println("Plot saved to $output_file")


	else
		println("Data does not have enough columns to plot.")
	end
end

plot_bench("cluster_tet4_adj_t125_f32_GH200.csv", "cluster_bench_plot.pdf")
