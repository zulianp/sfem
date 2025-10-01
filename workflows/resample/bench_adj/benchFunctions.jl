using CSV
using DataFrames
using Statistics
using PyPlot
using Distributions
using GZip
using Printf
using PrettyTables

function read_special_csv(filepath::String)
    df = DataFrame(CSV.File(filepath))
    return df
end

"""
	confidence_interval(data::Vector{T}, alpha::Float64 = 0.05) where T <: Real
Compute the (1 - alpha)*100% confidence interval for the mean of the data vector.
Uses the t-distribution for small sample sizes.
# Arguments
- `data::Vector{T}`: Vector of real numbers
- `alpha::Float64=0.05`: Significance level (default 0.05 for 95% CI)
# Returns
- Tuple `(lower_bound, upper_bound)` of the confidence interval
"""
function confidence_interval(data::Vector{T}, alpha::Float64=0.05) where {T<:Real}
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
                             t_val = quantile(TDist(n - 1), 0.95)  # 90% CI uses α/2 = 0.05, so 0.95 quantile
                             se * t_val
                         end) => :ci90_clock,
                         :Throughput => minimum => :min_Throughput,
                         :Throughput => maximum => :max_Throughput,
                         :Throughput => mean => :mean_Throughput,
                         :Throughput => (x -> std(x) / sqrt(length(x))) => :se_Throughput,
                         :Throughput => (x -> begin
                             n = length(x)
                             se = std(x) / sqrt(n)
                             t_val = quantile(TDist(n - 1), 0.95)  # 90% CI uses α/2 = 0.05, so 0.95 quantile
                             se * t_val
                         end) => :ci90_Throughput)

    return summary_df
end

"""
	col_with_max_Throughput(df::DataFrame)
Return the row with the highest mean_Throughput.
"""
function col_with_max_Throughput(df::DataFrame)
    max_row = df[argmax(df.:mean_Throughput), :]
    return max_row
end

"""
	cols_within_max_Throughput(df::DataFrame, top::Int = 10)
Return the top `top` rows with highest mean_Throughput.
"""
function cols_within_max_Throughput(df::DataFrame, top::Int=10)
    sorted_df = sort(df, :mean_Throughput; rev=true)
    return sorted_df[1:top, :]
end

"""
	plot_throughput_bars(df::DataFrame; logy::Bool = false, fpFormat::Int = 32)
Create a bar plot of mean Throughput with error bars for 90% confidence intervals.
# Arguments
- `df::DataFrame`: DataFrame with columns `cluster_size`, `tet_per_block`, `mean_Throughput`, `ci90_Throughput`
- `logy::Bool=false`: Whether to use a logarithmic scale for the y-axis
- `fpFormat::Int=32`: Floating point format (32 or 64) for title annotation
# Returns
- `fig`: PyPlot figure object
"""
function plot_throughput_bars(df::DataFrame;
                              logy::Bool=false,
                              fpFormat::Int=32,
                              fig_size=(10, 6),
                              fontsize=14,
                              plot_title=false)
    #
    required = ["cluster_size", "tet_per_block", "mean_Throughput", "ci90_Throughput"]
    missing = setdiff(required, names(df))
    if !isempty(missing)
        throw(ArgumentError("DataFrame is missing required columns: $(missing)"))
    end

    labels = ["($(r.cluster_size), $(r.tet_per_block))" for r in eachrow(df)]
    means = collect(skipmissing(df.mean_Throughput))
    errs = collect(skipmissing(df.ci90_Throughput))

    fig, ax = subplots(; figsize=fig_size)
    xs = 1:length(means)
    ax.bar(xs, means; color="lightblue", edgecolor="black")
    ax.errorbar(xs, means; yerr=errs, fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels; rotation=45, ha="right")
    ax.set_xlabel("(cluster size, TET / ThBlock)"; fontsize=fontsize)
    ax.set_ylabel("Mean TET/Sec"; fontsize=fontsize)

    ax.tick_params(; axis="both", which="major", labelsize=fontsize - 1)
    ax.tick_params(; axis="both", which="minor", labelsize=fontsize - 1)
    # Set the exponent fontsize specifically
    ax.yaxis.get_offset_text().set_fontsize(fontsize - 3)

    fp_string = fpFormat == 32 ? "f32" : fpFormat == 64 ? "f64" : "fp"
    if plot_title
        ax.set_title("Mean Throughput ± 90% CI ($fp_string)"; fontsize=fontsize)
    end

    if logy
        if any(means .<= 0)
            throw(ArgumentError("Cannot use log scale: mean_Throughput must be positive."))
        end
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(PyPlot.matplotlib.ticker.LogLocator(; base=10))
        ax.yaxis.set_minor_locator(PyPlot.matplotlib.ticker.LogLocator(; base=10,
                                                                       subs=collect(2:9)))
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
    results = Dict{String,DataFrame}()

    for filepath in filepaths
        if !isfile(filepath)
            @warn "File not found: $filepath"
            continue
        end

        try
            df = DataFrame(CSV.File(filepath))

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
function read_level_csvs(directory::String, pattern::String="*.csv")
    if !isdir(directory)
        throw(ArgumentError("Directory does not exist: $directory"))
    end

    filepaths = glob(pattern, directory)
    if isempty(filepaths)
        @warn "No files found matching pattern '$pattern' in directory '$directory'"
        return Dict{String,DataFrame}()
    end

    return read_level_csvs(filepaths)
end

"""
	combine_level_data(data_dict::Dict{String, DataFrame})

Combine multiple level DataFrames into a single DataFrame with a source column.
"""
function combine_level_data(data_dict::Dict{String,DataFrame})
    if isempty(data_dict)
        return DataFrame(; Source=String[], Level=Int[], Number_of_elements=Int[])
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
function plot_level_histograms(data_dict::Dict{String,DataFrame}; figsize=(10, 12))
    if length(data_dict) != 3
        throw(ArgumentError("Function expects exactly 3 datasets, got $(length(data_dict))"))
    end

    # Find maximum level across all datasets to determine number of bins
    max_level = 0
    for (_, df) in data_dict
        max_level = max(max_level, maximum(df.Level))
    end

    # Create figure with 3 subplots (3 rows, 1 column)
    fig, axes = subplots(3, 1; figsize=figsize)

    # Sort datasets by name for consistent ordering
    sorted_datasets = sort(collect(data_dict); by=x -> x[1])

    for (i, (filename, df)) in enumerate(sorted_datasets)
        ax = axes[i]

        # Create histogram with orientation="horizontal" for vertical classes
        counts, bins, patches = ax.hist(df.Level;
                                        bins=1:(max_level + 1),
                                        orientation="horizontal",
                                        alpha=0.7,
                                        edgecolor="black",
                                        color="skyblue")

        # Set labels and title
        ax.set_ylabel("Level")
        ax.set_xlabel("Frequency")
        ax.set_title("Level Distribution - $filename")

        # Set y-axis to show all levels
        ax.set_ylim(0.5, max_level + 0.5)
        ax.set_yticks(1:max_level)

        # Add grid for better readability
        ax.grid(true; alpha=0.3)

        # Add text annotation with total elements
        total_elements = sum(df.Number_of_elements)
        ax.text(0.02,
                0.98,
                "Total elements: $(total_elements)";
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=Dict("boxstyle" => "round,pad=0.3", "facecolor" => "white",
                          "alpha" => 0.8))
    end

    plt.tight_layout()
    return fig
end

"""
	plot_elements_by_level(data_dict::Dict{String, DataFrame}; figsize=(12, 8))

Create a line plot showing Number_of_elements vs Level for all datasets.
"""
function plot_elements_by_level(data_dict::Dict{String,DataFrame};
                                figsize=(12, 8),
                                fontsize=14,
                                legend_list=nothing)
    fig, ax = subplots(; figsize=figsize)

    colors = ["blue", "red", "green", "orange", "purple"]

    for (i, (filename, df)) in enumerate(data_dict)
        color = colors[mod(i - 1, length(colors)) + 1]

        my_label = isnothing(legend_list) ? filename : legend_list[i]

        ax.plot(df.Level, df.Number_of_elements;
                marker="o",
                label=my_label,
                color=color,
                linewidth=3,
                markersize=8)
    end

    ax.set_xlabel("Level (L)"; fontsize=fontsize)
    ax.set_ylabel("# Refined Elements"; fontsize=fontsize)
    # ax.set_title("Number of Refined Elements by Level"; fontsize=fontsize + 2)

    # Set tick label font sizes
    ax.tick_params(; axis="both", which="major", labelsize=fontsize - 1)
    ax.tick_params(; axis="both", which="minor", labelsize=fontsize - 1)

    # Set x-axis to show only integer values with step of 2
    min_level = minimum([minimum(df.Level) for (_, df) in data_dict])
    max_level = maximum([maximum(df.Level) for (_, df) in data_dict])
    x_ticks = collect(min_level:2:max_level)
    ax.set_xticks(x_ticks)

    legend_obj = ax.legend(; fontsize=fontsize, loc="best")
    legend_obj.get_frame().set_linewidth(0)  # Remove border line
    legend_obj.get_frame().set_facecolor("white")  # Optional: set background color
    legend_obj.get_frame().set_alpha(0.8)  # Optional: set transparency

    ax.grid(true; alpha=0.5)
    ax.set_yscale("log")  # Log scale often useful for element counts

    plt.tight_layout()
    return fig
end

"""
	read_alpha_data(filepath::String)
Read alpha values from a CSV file, handling gzipped files if necessary.
Returns a vector of alpha values.
"""
function read_alpha_data(filepath::String)
    if !isfile(filepath)
        throw(ArgumentError("File not found: $filepath"))
    end

    # Check if file is gzipped and decompress if needed
    if endswith(filepath, ".gz")
        # Read gzipped file
        df = DataFrame(CSV.File(GZip.read(filepath); header=false))
    else
        # Read regular file
        df = DataFrame(CSV.File(filepath; header=false))
    end

    # If it's a single column, extract it as a vector
    if ncol(df) == 1
        return Vector(df[!, 1])
    else
        # If multiple columns, flatten all data into a single vector
        return vec(Matrix(df))
    end
end

"""
	plot_alpha_histogram(alpha_vec::Vector; bins=100, figsize=(10, 6), 
						color="steelblue", alpha_val=0.7, show_stats=true)

Create a histogram of alpha values with optional statistical overlays.

# Arguments
- `alpha_vec::Vector`: Vector of alpha values to plot
- `bins::Int=100`: Number of histogram bins
- `figsize::Tuple=(10, 6)`: Figure size (width, height)
- `color::String="steelblue"`: Histogram color
- `alpha_val::Float64=0.7`: Transparency of histogram bars
- `show_stats::Bool=true`: Whether to show mean and std dev lines

# Returns
- `fig`: PyPlot figure object
"""
function plot_alpha_histogram(alpha_vec::Vector; bins=100, figsize=(10, 6),
                              color="steelblue", alpha_val=0.7, show_stats=true)
    # Create histogram using PyPlot
    fig, ax = subplots(; figsize=figsize)

    # Create the histogram
    n, bins_out, patches = ax.hist(alpha_vec; bins=bins, alpha=alpha_val,
                                   color=color, edgecolor="black")

    # Add labels and title
    ax.set_xlabel("Alpha Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Alpha Values")

    # Set scientific notation on y-axis
    ax.ticklabel_format(; style="scientific", axis="y", scilimits=(0, 0))

    if show_stats
        # Add statistics
        mean_val = mean(alpha_vec)
        std_val = std(alpha_vec)

        # Add vertical lines for mean and ±1 std dev
        ax.axvline(mean_val; color="red", linestyle="-", linewidth=2,
                   label="Mean $(round(mean_val, digits=2))")
        ax.axvline(mean_val - std_val; color="green", linestyle="--", linewidth=1,
                   label="±$(round(std_val, digits=2)) Std Dev")
        ax.axvline(mean_val + std_val; color="green", linestyle="--", linewidth=1)

        # Add legend
        ax.legend()
    end

    # Add grid
    ax.grid(true; alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    return fig
end

"""
	alpha_statistics_table(alpha_vectors::Vector{Vector{T}}, labels::Vector{String}) where T <: Real

Create a formatted table showing statistics for multiple alpha vectors.

# Arguments
- `alpha_vectors::Vector{Vector{T}}`: Vector of alpha data vectors
- `labels::Vector{String}`: Labels for each alpha vector (e.g., ["125", "250", "500"])

# Returns
- Displays a formatted table using PrettyTables
"""
function alpha_statistics_table(alpha_vectors::Vector{Vector{T}},
                                labels::Vector{String}) where {T<:Real}
    if length(alpha_vectors) != length(labels)
        throw(ArgumentError("Number of alpha vectors must match number of labels"))
    end

    # Calculate statistics for each vector
    stats_data = []

    for (i, alpha_vec) in enumerate(alpha_vectors)
        if isempty(alpha_vec)
            push!(stats_data, [labels[i], "mean", "median", "max", "min", "std", "count"])
        else
            push!(stats_data,
                  [labels[i],
                   mean(alpha_vec),
                   median(alpha_vec),
                   maximum(alpha_vec),
                   minimum(alpha_vec),
                   std(alpha_vec),
                   length(alpha_vec)])
        end
    end

    # Create DataFrame
    df = DataFrame(; Dataset=[row[1] for row in stats_data],
                   Mean=[row[2] for row in stats_data],
                   Median=[row[3] for row in stats_data],
                   Maximum=[row[4] for row in stats_data],
                   Minimum=[row[5] for row in stats_data],
                   StdDev=[row[6] for row in stats_data],
                   Count=[row[7] for row in stats_data])

    return df
end

"""
    save_figure_to_pdf(fig, filename::String; dpi=300, bbox_inches="tight")

Save a PyPlot figure to a PDF file.

# Arguments
- `fig`: PyPlot figure object to save
- `filename::String`: Output filename (with or without .pdf extension)
- `dpi::Int=300`: Resolution in dots per inch
- `bbox_inches::String="tight"`: Bounding box setting ("tight" removes extra whitespace)

# Returns
- `String`: Full path of the saved file

# Example
```julia
fig = plot_alpha_histogram(alpha_vec)
save_figure_to_pdf(fig, "alpha_distribution")
save_figure_to_pdf(fig, "results/alpha_hist.pdf", dpi=600)
```
"""
function save_figure_to_pdf(fig, filename::String; dpi=300, bbox_inches="tight")
    # Ensure .pdf extension
    if !endswith(filename, ".pdf")
        filename = filename * ".pdf"
    end

    # Create directory if it doesn't exist
    dir_path = dirname(filename)
    if !isempty(dir_path) && !isdir(dir_path)
        mkpath(dir_path)
    end

    # Save the figure
    fig.savefig(filename; format="pdf", dpi=dpi, bbox_inches=bbox_inches)

    println("Figure saved to: $filename")
    return abspath(filename)
end

"""
    save_multiple_figures_to_pdf(figures::Vector, filenames::Vector{String}; dpi=300)

Save multiple figures to PDF files.

# Arguments
- `figures::Vector`: Vector of PyPlot figure objects
- `filenames::Vector{String}`: Vector of output filenames
- `dpi::Int=300`: Resolution for all figures

# Example
```julia
figs = [plot_alpha_histogram(alpha_vec_125), plot_alpha_histogram(alpha_vec_250)]
names = ["alpha_125", "alpha_250"]
save_multiple_figures_to_pdf(figs, names)
```
"""
function save_multiple_figures_to_pdf(figures::Vector, filenames::Vector{String}; dpi=300)
    if length(figures) != length(filenames)
        throw(ArgumentError("Number of figures must match number of filenames"))
    end

    saved_files = String[]
    for (fig, filename) in zip(figures, filenames)
        saved_file = save_figure_to_pdf(fig, filename; dpi=dpi)
        push!(saved_files, saved_file)
    end

    return saved_files
end
