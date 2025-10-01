using CSV
using DataFrames
using Statistics
using PyPlot
using Distributions

function read_special_csv(filepath::String)
	df = CSV.File(filepath) |> DataFrame
	return df
end


function confidence_interval(data::Vector{T}, alpha::Float64 = 0.05) where T<:Real
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
