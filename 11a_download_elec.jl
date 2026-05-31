# 11a_download_elec.jl
# Download Victorian half-hourly electricity demand from tsibbledata GitHub

using Downloads, CSV, DataFrames, Statistics

mkpath("data")

const URL  = "https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/vic_elec/VIC2015/demand.csv"
const PATH = "data/vic_elec.csv"

if isfile(PATH)
    println("SKIP — already exists")
else
    println("Downloading vic_elec demand data...")
    Downloads.download(URL, PATH)
end

df = CSV.read(PATH, DataFrame)
println("Columns: ", names(df))
println("Rows: ", nrow(df))
println("First rows:")
println(first(df, 3))
