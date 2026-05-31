# 10a_download_sunspots.jl
# Download daily total sunspot numbers from SILSO (Royal Observatory of Belgium)
# Source: https://www.sidc.be/SILSO/datafiles
# Format: year, month, day, date_frac, SN_total, SN_std, n_obs, definitive

using Downloads
using CSV
using DataFrames
using Dates
using Statistics

mkpath("data")

const URL  = "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.txt"
const PATH = "data/sunspots_daily.csv"

if isfile(PATH)
    println("SKIP — data/sunspots_daily.csv already exists")
else
    println("Downloading daily sunspot numbers from SILSO...")
    raw_path = "data/sunspots_raw.txt"
    Downloads.download(URL, raw_path)

    # Parse fixed-width format: year month day date_frac SN std n_obs definitive
    rows = []
    for line in eachline(raw_path)
        isempty(strip(line)) && continue
        parts = split(strip(line))
        length(parts) < 5 && continue
        year  = parse(Int, parts[1])
        month = parse(Int, parts[2])
        day   = parse(Int, parts[3])
        sn    = parse(Float64, parts[5])   # total sunspot number
        sn < 0 && continue                 # -1 = missing
        push!(rows, (date=Date(year, month, day), sunspots=sn))
    end

    df = DataFrame(rows)
    sort!(df, :date)
    CSV.write(PATH, df)
    println("Saved $(nrow(df)) rows: $(first(df.date)) to $(last(df.date))")
    println("SN range: $(minimum(df.sunspots)) – $(maximum(df.sunspots))")
end

# Quick summary
df = CSV.read(PATH, DataFrame)
println("\nLoaded $(nrow(df)) daily observations")
println("Date range: $(first(df.date)) to $(last(df.date))")
println("Mean SN: $(round(mean(df.sunspots), digits=2))")
println("Zero days (solar minimum): $(sum(df.sunspots .== 0))")
