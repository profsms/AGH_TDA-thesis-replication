# 11b_elec_forecasting.jl
# Forecasting demonstration: topo features on Victorian half-hourly electricity demand
#
# Electricity demand has strong nonlinear intraday/intraweek dynamics.
# A lag-only model captures linear autocorrelation but not the nonlinear
# interactions between time-of-day, day-of-week, and temperature regime.
# The delay embedding of demand residuals (after weekly seasonal adjustment)
# should produce persistent H₁ generators corresponding to the nonlinear
# attractor structure of the demand cycle.
#
# Design:
#   Series:     vic_elec half-hourly demand (MW), Victoria 2012-2014
#   Transform:  Weekly seasonal differencing: r_t = y_t - y_{t-336}
#               removes intraday + intraweek cycle, leaves nonlinear residual
#   Baseline:   lags 1, 2, 48, 336 of the differenced series (strong benchmark)
#   Augmented:  baseline + topo_features(window=W, per-series τ and dim)
#   Window W:   336 half-hours (1 week)
#   Horizons:   h=1 (30 min ahead), h=48 (24 hours ahead)
#   CV:         expanding walk-forward, refit every K=20, test on last 2000 obs

using TopoTS
using XGBoost
using CSV
using DataFrames
using Statistics
using Printf
using Serialization
using Random

Random.seed!(42)

mkpath("checkpoints/elec")
mkpath("results")

# ── Load data ──────────────────────────────────────────────────────────────────

df = CSV.read("data/vic_elec.csv", DataFrame)
println("Columns: ", names(df))
println("Rows: ", nrow(df))
println("First 3 rows: ", first(df, 3))

# Column is "OperationalLessIndustrial", stored as String15 — parse to Float64
# Filter to 2012-2013 only (Excel date: 40909=2012-01-01, 41639=2013-12-31)
filter!(r -> r.Date >= 40909 && r.Date <= 41639, df)
println("After date filter (2012-2013): $(nrow(df)) rows")

demand_col = "OperationalLessIndustrial"
println("Using demand column: \'$demand_col\'")
y_raw = parse.(Float64, string.(df[!, demand_col]))
N     = length(y_raw)
println("N = $N half-hourly observations")
println("Demand range: $(minimum(y_raw)) – $(maximum(y_raw)) MW")

# ── Weekly seasonal differencing ───────────────────────────────────────────────
# S = 336 (48 half-hours × 7 days)
const S = 336
y = y_raw[S+1:end] .- y_raw[1:end-S]   # r_t = y_t - y_{t-S}
N = length(y)
println("\nAfter seasonal differencing: N = $N")
@printf("Residual stats: μ=%.2f  σ=%.2f  min=%.2f  max=%.2f\n",
        mean(y), std(y), minimum(y), maximum(y))

# ── Parameters ─────────────────────────────────────────────────────────────────

const LAG_LIST = [1, 2, 48, 336]   # strong electricity demand baseline
const Q        = length(LAG_LIST)
const W        = 336               # topo window: 1 week
const K        = 20
const N_TEST   = 2000
const H_LIST   = [1, 48]

base_spec = TopoFeatureSpec(
    dim_max            = 1,
    dim                = 3,
    lag                = 1,
    use_landscape      = true,
    n_landscape_layers = 2,
    n_landscape_grid   = 20,
    use_betti          = false,
    use_stats          = true,
    use_image          = false
)
p_topo     = length(feature_names(base_spec))
xgb_params = (num_round=150, max_depth=5, eta=0.05,
               subsample=0.8, colsample_bytree=0.8,
               objective="reg:squarederror", verbosity=0)

# ── Per-series embedding parameters ────────────────────────────────────────────

τ = optimal_lag(y)
d = clamp(optimal_dim(y; lag=τ, max_dim=6), 2, 6)
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
println("Embedding: τ=$τ, dim=$d  →  $p_topo topo features")

# ── Topo feature matrix ────────────────────────────────────────────────────────

n_windows = N - W + 1
ckpt_topo = "checkpoints/elec/F_topo.jls"

if isfile(ckpt_topo)
    println("\nLoading topo feature matrix from checkpoint...")
    F_topo, tgrid_land = deserialize(ckpt_topo)
    println("Loaded: $(size(F_topo))")
else
    println("\nComputing topo features ($n_windows windows)...")
    _dgms0 = persistent_homology(
                 embed(y[1:W]; dim=spec.dim, lag=spec.lag);
                 dim_max=spec.dim_max, filtration=spec.filtration,
                 threshold=spec.threshold)
    _λ0        = TopoTS.Landscapes.landscape(_dgms0, 0;
                     n_grid=spec.n_landscape_grid,
                     n_layers=spec.n_landscape_layers)
    tgrid_land = _λ0.tgrid

    F_topo = Matrix{Float64}(undef, n_windows, p_topo)
    for i in 1:n_windows
        i % 5000 == 1 && @printf("  window %d/%d\n", i, n_windows)
        F_topo[i, :] = topo_features(y[i:i+W-1]; spec=spec,
                                      tgrid_landscape=tgrid_land,
                                      tgrid_betti=nothing)
    end
    F_topo[isnan.(F_topo) .| isinf.(F_topo)] .= 0.0
    serialize(ckpt_topo, (F_topo, tgrid_land))
    println("Done. Checkpointed.")
end

# ── Walk-forward ───────────────────────────────────────────────────────────────
# Alignment: topo row i → window y[i:i+W-1], endpoint i+W-1
# Lag features at endpoint: y[i+W-1-lag] for lag in LAG_LIST
# Target at horizon h: y[i+W-1+h]

σ = std(y)
results = Dict{Int, NamedTuple}()

for h in H_LIST
    n_valid    = n_windows - h
    test_start = max(n_valid - N_TEST + 1, maximum(LAG_LIST) + 1)
    n_test_act = n_valid - test_start + 1

    println("\n", "="^55)
    println("Horizon h=$h  ($(h*30) min ahead)  |  test: $n_test_act points")
    println("="^55)

    ae_base = Float64[]; ae_aug = Float64[]
    bst_b   = nothing;   bst_a  = nothing

    # Lag feature for topo index i: y at endpoint minus lag offset
    # Minimum safe topo index: need i+W-1-max_lag >= 1  =>  i >= max_lag-W+2
    # With W=336 and max_lag=336: i >= 1. But lag=336 means y[i+W-1-336]=y[i-1]
    # so i >= 2 is required. Use i_min = maximum(LAG_LIST) - W + 2
    i_min = max(1, maximum(LAG_LIST) - W + 2)
    make_lag_row(i) = [y[i+W-1-l] for l in LAG_LIST]'

    for t_test in test_start:n_valid
        t_train = t_test - 1

        if isnothing(bst_b) || (t_test - test_start) % K == 0
            X_lag_tr = reduce(vcat, make_lag_row(i) for i in i_min:t_train)
            X_top_tr = F_topo[i_min:t_train, :]
            y_tr     = [y[i+W-1+h] for i in i_min:t_train]
            X_aug_tr = hcat(X_lag_tr, X_top_tr)
            bst_b    = xgboost((X_lag_tr, y_tr); xgb_params...)
            bst_a    = xgboost((X_aug_tr, y_tr); xgb_params...)
        end

        X_lag_te = make_lag_row(t_test)
        X_aug_te = hcat(X_lag_te, F_topo[t_test:t_test, :])
        y_te     = y[t_test+W-1+h]

        push!(ae_base, abs(XGBoost.predict(bst_b, X_lag_te)[1] - y_te))
        push!(ae_aug,  abs(XGBoost.predict(bst_a, X_aug_te)[1] - y_te))
    end

    mae_b  = mean(ae_base) / σ;   mae_a  = mean(ae_aug)  / σ
    rmse_b = sqrt(mean(ae_base .^ 2)) / σ
    rmse_a = sqrt(mean(ae_aug  .^ 2)) / σ
    Δmae   = (mae_b - mae_a) / mae_b * 100
    Δrmse  = (rmse_b - rmse_a) / rmse_b * 100

    results[h] = (mae_b=mae_b, mae_a=mae_a, rmse_b=rmse_b, rmse_a=rmse_a,
                  Δmae=Δmae, Δrmse=Δrmse, n=n_test_act)

    @printf("Baseline   MAE=%.4f  RMSE=%.4f\n", mae_b, rmse_b)
    @printf("Augmented  MAE=%.4f  RMSE=%.4f\n", mae_a, rmse_a)
    @printf("Δ_MAE=%+.2f%%  Δ_RMSE=%+.2f%%  → %s\n",
            Δmae, Δrmse,
            Δmae > 2  ? "TOPO HELPS ✓" :
            Δmae < -2 ? "TOPO HURTS"   : "NEUTRAL")
end

# ── Save & summarise ───────────────────────────────────────────────────────────

serialize("checkpoints/elec/results.jls", results)
csv_rows = [(h=h, mae_b=results[h].mae_b, mae_a=results[h].mae_a,
             Δmae=results[h].Δmae, rmse_b=results[h].rmse_b,
             rmse_a=results[h].rmse_a, Δrmse=results[h].Δrmse,
             n=results[h].n) for h in H_LIST]
CSV.write("results/elec_forecasting.csv", DataFrame(csv_rows))

println("\n", "="^55)
println("FINAL SUMMARY  (σ=$(round(σ,digits=2)) MW, normalised)")
println("Series: Victorian half-hourly electricity demand")
println("Transform: weekly seasonal differencing (S=336)")
println("Embedding: τ=$τ, dim=$d  |  W=$W  |  lags=$(LAG_LIST)")
println("="^55)
@printf("%-4s  %-12s  %7s  %7s  %7s  %7s\n",
        "h", "Model", "MAE", "RMSE", "ΔMAE%", "ΔRMSE%")
println("-"^55)
for h in H_LIST
    r = results[h]
    @printf("%-4d  %-12s  %7.4f  %7.4f\n",          h, "Baseline",  r.mae_b, r.rmse_b)
    @printf("%-4s  %-12s  %7.4f  %7.4f  %+6.2f%%  %+6.2f%%\n",
            "", "Augmented", r.mae_a, r.rmse_a, r.Δmae, r.Δrmse)
    println("-"^55)
end
println("Results saved to results/elec_forecasting.csv")
