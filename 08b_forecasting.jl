# 08b_forecasting.jl
# Multi-series forecasting benchmark: baseline lags vs. baseline + topo features
# Series:    NG, VIX, Gold, TNX, BTC, Crude Oil
# Horizons:  h = 1 and h = 5
# Model:     XGBoost (100 rounds, max_depth=4, eta=0.05, subsample=0.8)
# CV:        Expanding-window walk-forward, refit every K=20 steps
#            Test window: last 500 observations per series
# Features:  Baseline — last Q=5 log-returns
#            Augmented — baseline + topo_features (window W=252, shared grids)

using TopoTS
using XGBoost
using CSV
using DataFrames
using Statistics
using Printf
using Dates
using Random
using Serialization

Random.seed!(42)

# ── Parameters ─────────────────────────────────────────────────────────────────

const Q         = 5      # lag features
const W         = 252    # topo window (1 trading year)
const K         = 20     # refit every K steps
const N_TEST    = 500    # test points per series
const H_LIST    = [1, 5]

# Base spec — lag and dim are set per series using optimal_lag / optimal_dim
base_spec = TopoFeatureSpec(
    dim_max            = 1,
    dim                = 3,      # overridden per series
    lag                = 1,      # overridden per series
    use_landscape      = true,
    n_landscape_layers = 2,
    n_landscape_grid   = 20,
    use_betti          = false,
    use_stats          = true,
    use_image          = false
)
# Feature count is fixed by everything except lag/dim (which don't affect vector length)
p_topo = length(feature_names(base_spec))
xgb_params = (num_round=100, max_depth=4, eta=0.05,
               subsample=0.8, objective="reg:squarederror", verbosity=0)

mkpath("checkpoints/forecasting")
mkpath("results")

series_list = [
    ("ng",     "Nat Gas (NG=F)"),
    ("vix",    "VIX"),
    ("gold",   "Gold (GC=F)"),
    ("tnx",    "10Y Treasury (TNX)"),
    ("btc",    "Bitcoin (BTC)"),
    ("crude",  "Crude Oil (CL=F)"),
    ("nasdaq", "NASDAQ (IXIC)"),
    ("nikkei", "Nikkei 225"),
    ("hsi",    "Hang Seng (HSI)"),
    ("copper", "Copper (HG=F)"),
    ("wheat",  "Wheat (ZW=F)"),
    ("silver", "Silver (SI=F)"),
    ("eurusd", "EUR/USD"),
    ("usdjpy", "USD/JPY"),
    ("vvix",   "VVIX"),
]

# ── Result storage ─────────────────────────────────────────────────────────────
# results[slug][h] = (mae_b, mae_a, rmse_b, rmse_a, Δmae, Δrmse, n_test)

all_results = Dict{String, Dict}()

# ── Per-series loop ────────────────────────────────────────────────────────────

for (slug, label) in series_list
    path = "data/$(slug).csv"
    if !isfile(path)
        println("SKIP $label — file not found: $path")
        continue
    end

    df      = CSV.read(path, DataFrame)
    sort!(df, :date)
    prices  = df.close
    returns = diff(log.(prices))
    N       = length(returns)
    σ       = std(returns)

    if N < W + N_TEST + maximum(H_LIST) + Q
        println("SKIP $label — insufficient data (N=$N)")
        continue
    end

    # Per-series embedding parameters (AMI lag, FNN dimension)
    τ = optimal_lag(returns)
    d = optimal_dim(returns; lag=τ, max_dim=5)
    d = clamp(d, 2, 5)   # guard against degenerate estimates

    spec = TopoFeatureSpec(
        dim_max            = base_spec.dim_max,
        dim                = d,
        lag                = τ,
        filtration         = base_spec.filtration,
        threshold          = base_spec.threshold,
        use_landscape      = base_spec.use_landscape,
        n_landscape_layers = base_spec.n_landscape_layers,
        n_landscape_grid   = base_spec.n_landscape_grid,
        use_betti          = base_spec.use_betti,
        n_betti_grid       = base_spec.n_betti_grid,
        use_stats          = base_spec.use_stats,
        use_image          = base_spec.use_image,
        n_image_pixels     = base_spec.n_image_pixels,
    )

    println("\n", "="^62)
    println("Series: $label  (N=$N, σ=$(round(σ,digits=5)), τ=$τ, dim=$d)")
    println("="^62)

    # ── Topo feature matrix ────────────────────────────────────────────────────
    # F_topo[i,:] uses returns[i:i+W-1], i=1..n_windows

    n_windows = N - W + 1
    println("Computing topo features ($n_windows windows)...")

    # Shared grids from first window
    _dgms0 = persistent_homology(
                 embed(returns[1:W]; dim=spec.dim, lag=spec.lag);
                 dim_max=spec.dim_max, filtration=spec.filtration,
                 threshold=spec.threshold)
    _λ0        = TopoTS.Landscapes.landscape(_dgms0, 0;
                     n_grid=spec.n_landscape_grid,
                     n_layers=spec.n_landscape_layers)
    tgrid_land = _λ0.tgrid

    F_topo = Matrix{Float64}(undef, n_windows, p_topo)
    for i in 1:n_windows
        i % 1000 == 1 && @printf("  window %d/%d\n", i, n_windows)
        F_topo[i, :] = topo_features(returns[i:i+W-1]; spec=spec,
                                      tgrid_landscape=tgrid_land,
                                      tgrid_betti=nothing)
    end
    F_topo[isnan.(F_topo) .| isinf.(F_topo)] .= 0.0
    println("Topo features done.")

    # ── Walk-forward ───────────────────────────────────────────────────────────
    # Alignment: topo row i → endpoint i+W-1, target at i+W-1+h
    # Valid i: 1..n_windows-h
    # Test:    last N_TEST valid indices
    # Train:   all valid indices before test start (expanding)

    all_results[slug] = Dict()

    for h in H_LIST
        n_valid    = n_windows - h
        test_start = max(n_valid - N_TEST + 1, Q + 1)
        n_test_act = n_valid - test_start + 1

        println("\n  h=$h  |  test indices $test_start..$n_valid  ($n_test_act points)")

        ae_base = Float64[]
        ae_aug  = Float64[]

        bst_b = nothing; bst_a = nothing   # cached models
        make_lag(i) = returns[i+W-Q : i+W-1]

        for t_test in test_start:n_valid
            t_train = t_test - 1   # train on 1..t_train

            # Refit every K steps (or first step)
            if isnothing(bst_b) || (t_test - test_start) % K == 0
                X_lag_tr = reduce(vcat, make_lag(i)' for i in 1:t_train)
                X_top_tr = F_topo[1:t_train, :]
                y_tr     = [returns[i+W-1+h] for i in 1:t_train]
                X_aug_tr = hcat(X_lag_tr, X_top_tr)

                bst_b = xgboost((X_lag_tr, y_tr); xgb_params...)
                bst_a = xgboost((X_aug_tr, y_tr); xgb_params...)
            end

            # Test point
            X_lag_te = make_lag(t_test)'
            X_aug_te = hcat(X_lag_te, F_topo[t_test:t_test, :])
            y_te     = returns[t_test+W-1+h]

            push!(ae_base, abs(XGBoost.predict(bst_b, X_lag_te)[1] - y_te))
            push!(ae_aug,  abs(XGBoost.predict(bst_a, X_aug_te)[1] - y_te))
        end

        mae_b  = mean(ae_base) / σ;   mae_a  = mean(ae_aug)  / σ
        rmse_b = sqrt(mean(ae_base .^ 2)) / σ
        rmse_a = sqrt(mean(ae_aug  .^ 2)) / σ
        Δmae   = (mae_b - mae_a)  / mae_b  * 100
        Δrmse  = (rmse_b - rmse_a) / rmse_b * 100

        all_results[slug][h] = (mae_b=mae_b, mae_a=mae_a,
                                 rmse_b=rmse_b, rmse_a=rmse_a,
                                 Δmae=Δmae, Δrmse=Δrmse, n=n_test_act)

        @printf("  Baseline   MAE=%.4f  RMSE=%.4f\n", mae_b, rmse_b)
        @printf("  Augmented  MAE=%.4f  RMSE=%.4f\n", mae_a, rmse_a)
        @printf("  Δ_MAE=%+.2f%%  Δ_RMSE=%+.2f%%\n", Δmae, Δrmse)
    end
    # Checkpoint after each series
    serialize("checkpoints/forecasting/$(slug).jls", all_results[slug])
end

# ── Save results to CSV ────────────────────────────────────────────────────────

mkpath("results")
csv_rows = []
for (slug, label) in series_list
    haskey(all_results, slug) || continue
    for h in H_LIST
        haskey(all_results[slug], h) || continue
        r = all_results[slug][h]
        push!(csv_rows, (
            series=label, slug=slug, h=h,
            mae_baseline=r.mae_b, mae_augmented=r.mae_a, delta_mae_pct=r.Δmae,
            rmse_baseline=r.rmse_b, rmse_augmented=r.rmse_a, delta_rmse_pct=r.Δrmse,
            n_test=r.n
        ))
    end
end
CSV.write("results/financial_forecasting.csv", DataFrame(csv_rows))
println("Results saved to results/financial_forecasting.csv")

# ── Summary table ──────────────────────────────────────────────────────────────

println("\n\n", "="^90)
println("BENCHMARK SUMMARY")
println("="^90)
@printf("%-20s  %-6s  %7s  %7s  %7s  %7s  %7s  %7s  %6s\n",
        "Series", "h",
        "MAE-B", "MAE-A", "ΔMAE%",
        "RMSE-B", "RMSE-A", "ΔRMSE%", "n_test")
println("-"^90)

for (slug, label) in series_list
    haskey(all_results, slug) || continue
    first_h = true
    for h in H_LIST
        haskey(all_results[slug], h) || continue
        r = all_results[slug][h]
        sname = first_h ? label : ""
        first_h = false
        winner = r.Δmae > 2 ? "↑" : r.Δmae < -2 ? "↓" : "="
        @printf("%-20s  %-6d  %7.4f  %7.4f  %+6.2f%% %s  %7.4f  %7.4f  %+6.2f%%  %6d\n",
                sname, h,
                r.mae_b, r.mae_a, r.Δmae, winner,
                r.rmse_b, r.rmse_a, r.Δrmse, r.n)
    end
    println("-"^90)
end
println("="^90)
println("↑ = topo helps (Δ>2%)  ↓ = topo hurts (Δ<-2%)  = = neutral")
