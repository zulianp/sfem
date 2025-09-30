using CSV
using DataFrames
using Statistics

function read_special_csv(filepath::String)
	df = CSV.File(filepath) |> DataFrame
	return df
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
		:Throughput => minimum => :min_Throughput,
		:Throughput => maximum => :max_Throughput,
		:Throughput => mean => :mean_Throughput,
		:Throughput => (x -> std(x) / sqrt(length(x))) => :se_Throughput,
	)

	return summary_df

end


function col_with_max_Throughput(df::DataFrame)
    max_row = df[argmax(df.:mean_Throughput), :]
    return max_row
end

function cols_within_max_Throughput(df::DataFrame, top::Int=10)
    sorted_df = sort(df, :mean_Throughput, rev=true)
    return sorted_df[1:top, :]
end