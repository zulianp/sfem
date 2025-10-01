using CSV
using DataFrames
using Statistics
using PyPlot
using Distributions

function read_special_csv(filepath::String)
	df = CSV.File(filepath) |> DataFrame
	return df
end


function confidence_interval(data::Vector{T}, alpha::Float64 = 0.05) where T <: Real
	n = length(data)
	if n < 2
		throw(ArgumentError("At least two data points are required to compute a confidence interval."))
	end
	mean_val = mean(data)
	std_err = std(data) / sqrt(n)
	t_dist = TDist(n - 1)
	t_val = quantile(t_dist, 1 - alpha / 2)
	margin_of_error = t_val * std_err
	return (mean_val - margin_of_error, mean_val + margin_of_error)
end

"""
	summarize_cluster_stats(df::DataFrame)

Group rows by `cluster_size` and `tet_per_block`, compute mean and standard error
for `clock` and `Throughput`, and return a DataFrame with one row per group.
"""
function summarize_cluster_stats(df::DataFrame)
	required_cols = ["cluster_size", "tet_per_block", "nelements", "clock", "Throughput"]
	missing_cols = setdiff(required_cols, names(df))
	if !isempty(missing_cols)
		throw(ArgumentError("DataFrame I is missing required columns: $(missing_cols)"))
	end

	grouped = groupby(my_df, [:cluster_size, :tet_per_block])

	summary_df = combine(grouped,
		:clock => minimum => :min_clock,
		:clock => maximum => :max_clock,
		:clock => mean => :mean_clock,
		:clock => (x -> std(x) / sqrt(length(x))) => :se_clock,
		:clock => (x -> begin
			n = length(x)
			se = std(x) / sqrt(n)
			t_val = quantile(TDist(n-1), 0.95)  # 90% CI uses α/2 = 0.05, so 0.95 quantile
			se * t_val
		end) => :ci90_clock,
		:Throughput => minimum => :min_Throughput,
		:Throughput => maximum => :max_Throughput,
		:Throughput => mean => :mean_Throughput,
		:Throughput => (x -> std(x) / sqrt(length(x))) => :se_Throughput,
		:Throughput => (x -> begin
			n = length(x)
			se = std(x) / sqrt(n)
			t_val = quantile(TDist(n-1), 0.95)  # 90% CI uses α/2 = 0.05, so 0.95 quantile
			se * t_val
		end) => :ci90_Throughput,
	)

	return summary_df

end


function col_with_max_Throughput(df::DataFrame)
	max_row = df[argmax(df.:mean_Throughput), :]
	return max_row
end

function cols_within_max_Throughput(df::DataFrame, top::Int = 10)
	sorted_df = sort(df, :mean_Throughput, rev = true)
	return sorted_df[1:top, :]
end

function plot_throughput_bars(df::DataFrame; logy::Bool = false, fpFormat::Int = 32)
	required = ["cluster_size", "tet_per_block", "mean_Throughput", "ci90_Throughput"]
	missing = setdiff(required, names(df))
	if !isempty(missing)
		throw(ArgumentError("DataFrame is missing required columns: $(missing)"))
	end

	labels = ["($(r.cluster_size), $(r.tet_per_block))" for r in eachrow(df)]
	means = collect(skipmissing(df.mean_Throughput))
	errs = collect(skipmissing(df.ci90_Throughput))

	fig, ax = subplots()
	xs = 1:length(means)
	ax.bar(xs, means, color = "lightblue", edgecolor = "black")
	ax.errorbar(xs, means, yerr = errs, fmt = "none", ecolor = "black", capsize = 4)
	ax.set_xticks(xs)
	ax.set_xticklabels(labels, rotation = 45, ha = "right")
	ax.set_xlabel("(cluster size, TET / ThBlock)")
	ax.set_ylabel("Mean Throughput")

	fp_string = fpFormat == 32 ? "f32" : fpFormat == 64 ? "f64" : "fp"
	ax.set_title("Mean Throughput ± 90% CI ($fp_string)")
	if logy
		if any(means .<= 0)
			throw(ArgumentError("Cannot use log scale: mean_Throughput must be positive."))
		end
		ax.set_yscale("log")
		ax.yaxis.set_major_locator(PyPlot.matplotlib.ticker.LogLocator(base = 10))
		ax.yaxis.set_minor_locator(PyPlot.matplotlib.ticker.LogLocator(base = 10, subs = collect(2:9)))
		ax.yaxis.set_minor_formatter(PyPlot.matplotlib.ticker.NullFormatter())
	end

	fig.tight_layout()
	return fig
end


"""
	read_level_csvs(filepaths::Vector{String})

Read multiple CSV files containing Level and Number_of_elements columns.
Returns a dictionary where keys are filenames and values are DataFrames.
"""
function read_level_csvs(filepaths::Vector{String})
	results = Dict{String, DataFrame}()

	for filepath in filepaths
		if !isfile(filepath)
			@warn "File not found: $filepath"
			continue
		end

		try
			df = CSV.File(filepath) |> DataFrame

			# Validate required columns
			required_cols = ["Level", "Number_of_elements"]
			missing_cols = setdiff(required_cols, names(df))

			if !isempty(missing_cols)
				@warn "File $filepath missing required columns: $missing_cols"
				continue
			end

			# Store with filename as key
			filename = basename(filepath)
			results[filename] = df

		catch e
			@warn "Error reading $filepath: $e"
		end
	end

	return results
end

"""
	read_level_csvs(directory::String, pattern::String = "*.csv")

Read all CSV files matching a pattern from a directory.
"""
function read_level_csvs(directory::String, pattern::String = "*.csv")
	if !isdir(directory)
		throw(ArgumentError("Directory does not exist: $directory"))
	end

	filepaths = glob(pattern, directory)
	if isempty(filepaths)
		@warn "No files found matching pattern '$pattern' in directory '$directory'"
		return Dict{String, DataFrame}()
	end

	return read_level_csvs(filepaths)
end

"""
	combine_level_data(data_dict::Dict{String, DataFrame})

Combine multiple level DataFrames into a single DataFrame with a source column.
"""
function combine_level_data(data_dict::Dict{String, DataFrame})
	if isempty(data_dict)
		return DataFrame(Source = String[], Level = Int[], Number_of_elements = Int[])
	end

	combined_dfs = DataFrame[]

	for (filename, df) in data_dict
		df_copy = copy(df)
		df_copy.Source = fill(filename, nrow(df))
		push!(combined_dfs, df_copy)
	end

	return vcat(combined_dfs...)
end

"""
	plot_level_histograms(data_dict::Dict{String, DataFrame}; figsize=(10, 12))

Create a 3x1 subplot with histograms showing Level distribution for each dataset.
Classes (Level) are on the vertical axis, frequency on horizontal axis.
Number of bins is determined by the maximum Level across all datasets.
"""
function plot_level_histograms(data_dict::Dict{String, DataFrame}; figsize = (10, 12))
	if length(data_dict) != 3
		throw(ArgumentError("Function expects exactly 3 datasets, got $(length(data_dict))"))
	end

	# Find maximum level across all datasets to determine number of bins
	max_level = 0
	for (_, df) in data_dict
		max_level = max(max_level, maximum(df.Level))
	end

	# Create figure with 3 subplots (3 rows, 1 column)
	fig, axes = subplots(3, 1, figsize = figsize)

	# Sort datasets by name for consistent ordering
	sorted_datasets = sort(collect(data_dict), by = x->x[1])

	for (i, (filename, df)) in enumerate(sorted_datasets)
		ax = axes[i]

		# Create histogram with orientation="horizontal" for vertical classes
		counts, bins, patches = ax.hist(df.Level,
			bins = 1:(max_level+1),
			orientation = "horizontal",
			alpha = 0.7,
			edgecolor = "black",
			color = "skyblue")

		# Set labels and title
		ax.set_ylabel("Level")
		ax.set_xlabel("Frequency")
		ax.set_title("Level Distribution - $filename")

		# Set y-axis to show all levels
		ax.set_ylim(0.5, max_level + 0.5)
		ax.set_yticks(1:max_level)

		# Add grid for better readability
		ax.grid(true, alpha = 0.3)

		# Add text annotation with total elements
		total_elements = sum(df.Number_of_elements)
		ax.text(0.02, 0.98, "Total elements: $(total_elements)", 
            transform = ax.transAxes, 
            verticalalignment = "top",
            bbox = Dict("boxstyle" => "round,pad=0.3", "facecolor" => "white", "alpha" => 0.8))
	end

	plt.tight_layout()
	return fig
end

"""
	plot_elements_by_level(data_dict::Dict{String, DataFrame}; figsize=(12, 8))

Create a line plot showing Number_of_elements vs Level for all datasets.
"""
function plot_elements_by_level(data_dict::Dict{String, DataFrame}; figsize = (12, 8))
	fig, ax = subplots(figsize = figsize)

	colors = ["blue", "red", "green", "orange", "purple"]

	for (i, (filename, df)) in enumerate(data_dict)
		color = colors[mod(i-1, length(colors))+1]
		ax.plot(df.Level, df.Number_of_elements,
			marker = "o", label = filename, color = color, linewidth = 2)
	end

	ax.set_xlabel("Level")
	ax.set_ylabel("Number of Refined Elements")
	ax.set_title("Number of Refined Elements by Level")
	ax.legend()
	ax.grid(true, alpha = 0.3)
	ax.set_yscale("log")  # Log scale often useful for element counts

	plt.tight_layout()
	return fig
end
