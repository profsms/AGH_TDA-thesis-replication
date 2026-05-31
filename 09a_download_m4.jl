# 09a_download_m4.jl
# Download M4 daily training and test sets from official GitHub
# and extract first 50 daily series with sufficient length

using CSV
using DataFrames
using Downloads
using Printf

mkpath("data/m4")

const TRAIN_URL = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Daily-train.csv"
const TEST_URL  = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Daily-test.csv"
const INFO_URL  = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/M4-info.csv"

function fetch_csv(url, path; silencewarn=false)
    if isfile(path)
        println("SKIP $(basename(path)) (already exists)")
    else
        println("Downloading $(basename(path))...")
        Downloads.download(url, path)
    end
    return CSV.read(path, DataFrame; silencewarnings=silencewarn)
end

train_df = fetch_csv(TRAIN_URL, "data/m4/Daily-train.csv"; silencewarn=true)
test_df  = fetch_csv(TEST_URL,  "data/m4/Daily-test.csv";  silencewarn=true)
info_df  = fetch_csv(INFO_URL,  "data/m4/M4-info.csv")

println("\nTrain: $(nrow(train_df)) series")
println("Test:  $(nrow(test_df)) series")

# Inspect info columns and unique frequency values
println("Info columns: ", names(info_df))
freq_col = names(info_df)[findfirst(n -> occursin("req", lowercase(n)), names(info_df))]
println("Frequency column: '$freq_col'")
println("Unique values: ", unique(info_df[!, freq_col]))

# Series IDs are in column V1 of train_df — all daily series start with "D"
id_col = "V1"
println("\nFirst 5 IDs: ", train_df[1:5, id_col])

# Extract and save first 50 daily series with >= 800 observations
const MIN_OBS  = 800
const N_SERIES = 50

println("\nExtracting $N_SERIES daily series with >= $MIN_OBS observations...")

let saved = 0, skipped = 0

for row in eachrow(train_df)
    saved >= N_SERIES && break
    sid = string(row[id_col])
    startswith(sid, "D") || continue

    # Extract non-missing numeric values (ragged rows → trailing missings)
    vals = Float64[]
    for c in names(train_df)[2:end]
        v = row[c]
        (ismissing(v) || !(v isa Number)) && break
        push!(vals, Float64(v))
    end

    if length(vals) < MIN_OBS
        skipped += 1
        continue
    end

    # Get test values
    test_rows = filter(r -> string(r[id_col]) == sid, test_df)
    test_vals = Float64[]
    if nrow(test_rows) > 0
        r = first(eachrow(test_rows))
        for c in names(test_df)[2:end]
            v = r[c]
            (ismissing(v) || !(v isa Number)) && break
            push!(test_vals, Float64(v))
        end
    end

    all_vals = vcat(vals, test_vals)
    df_out = DataFrame(
        t     = 1:length(all_vals),
        value = all_vals,
        split = vcat(fill("train", length(vals)), fill("test", length(test_vals)))
    )
    CSV.write("data/m4/$(sid).csv", df_out)
    saved += 1
    @printf("  Saved %s: %d train + %d test = %d total\n",
            sid, length(vals), length(test_vals), length(all_vals))
end

println("\nSaved $saved series, skipped $skipped (too short)")
println("Files in data/m4/D*.csv")
end  # let
