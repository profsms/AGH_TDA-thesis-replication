# 08a_download_all.jl
# Download daily closes for all benchmark series via MarketData.jl
# Skips download if file already exists.

using MarketData
using TimeSeries
using CSV
using DataFrames
using Dates
using Printf

mkpath("data")

# Panel of 15 series covering diverse asset classes and dynamics:
# Energy, metals, rates, volatility, crypto, equity indices, FX, agriculture
tickers = [
    # Already in panel
    ("NG=F",   "ng",      "Nat Gas (Henry Hub)"),
    ("^VIX",   "vix",     "VIX (S&P 500 vol)"),
    ("GC=F",   "gold",    "Gold futures"),
    ("^TNX",   "tnx",     "10Y Treasury yield"),
    ("BTC-USD","btc",     "Bitcoin USD"),
    ("CL=F",   "crude",   "Crude Oil WTI"),
    # Equity indices — regime-switching, nonlinear
    ("^IXIC",  "nasdaq",  "NASDAQ Composite"),
    ("^N225",  "nikkei",  "Nikkei 225"),
    ("^HSI",   "hsi",     "Hang Seng Index"),
    # Commodities — seasonality + supply shocks
    ("HG=F",   "copper",  "Copper futures"),
    ("ZW=F",   "wheat",   "Wheat futures"),
    ("SI=F",   "silver",  "Silver futures"),
    # FX — carry dynamics, mean-reversion
    ("EURUSD=X","eurusd", "EUR/USD"),
    ("JPY=X",  "usdjpy",  "USD/JPY"),
    # Alternative volatility
    ("^VVIX",  "vvix",    "VVIX (vol of VIX)"),
]

function download_series(ticker, slug, label)
    path = "data/$(slug).csv"
    if isfile(path)
        df = CSV.read(path, DataFrame)
        @printf("  SKIP %-28s (already have %d rows)\n", label, nrow(df))
        return
    end
    try
        ta = yahoo(ticker, YahooOpt(
            period1  = DateTime(2000, 1, 1),
            period2  = DateTime(2024, 12, 31),
            interval = "1d"
        ))
        closes     = ta[:Close]
        raw_dates  = timestamp(closes)
        raw_values = values(closes)[:, 1]
        df = DataFrame(
            date  = raw_dates,
            close = [v isa Number ? Float64(v) : NaN for v in raw_values]
        )
        filter!(r -> !isnan(r.close) && r.close > 0, df)
        sort!(df, :date)
        CSV.write(path, df)
        @printf("  ✓  %-28s  %d rows  %s – %s\n",
                label, nrow(df), first(df.date), last(df.date))
    catch e
        @printf("  ✗  %-28s  FAILED: %s\n", label, e)
    end
end

println("Downloading $(length(tickers)) series...\n")
for (ticker, slug, label) in tickers
    download_series(ticker, slug, label)
end
println("\nDone. Files in data/")
