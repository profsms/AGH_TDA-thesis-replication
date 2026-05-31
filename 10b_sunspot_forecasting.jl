# 10b_sunspot_forecasting.jl
# Forecasting demonstration: topo features on daily sunspot numbers
#
# Sunspot numbers exhibit a well-known ~11-year (Schwabe) cycle whose delay
# embedding produces a persistent H₁ generator — the theoretical conditions
# for topological features to carry predictive information are satisfied by
# construction (cf. Ch 5, Takens 1981).
#
# Design:
#   - Series: SILSO daily total sunspot number (SN_d_tot_V2.0)
#   - Transformation: sqrt(SN + 1) for variance stabilisation (SN is
#     count-like, right-skewed, with many zeros at solar minimum)
#   - Features: baseline = last Q=27 days (one solar rotation); augmented =
#     baseline + topo_features with per-series optimal_lag / optimal_dim
#   - Window W: 365 days (one year) — shorter than the solar cycle but long
#     enough for meaningful PH; the H₁ generator from the ~11yr cycle is
#     visible even in yearly windows as a large-persistence point
#   - Horizons: h = 1 and h = 27 (one solar rotation ahead)
#   - CV: expanding walk-forward, refit every K=20 steps, test on last 1000 obs
#   - Model: XGBoost

using TopoTS
using XGBoost
using CSV
using DataFrames
using Statistics
using Printf
using Dates
using Serialization
using Random

Random.seed!(42)

mkpath("checkpoints/sunspots")
mkpath("results")

# ── Load and transform ─────────────────────────────────────────────────────────

df      = CSV.read("data/sunspots_daily.csv", DataFrame)
sort!(df, :date)

# Variance-stabilising transform: sqrt(SN + 1)
# Inverse: SN_hat = max(0, y_hat^2 - 1)
y_raw = df.sunspots
y     = sqrt.(y_raw .+ 1.0)
N     = length(y)
dates = df.date

println("Loaded $N daily sunspot observations")
println("Date range: $(first(dates)) to $(last(dates))")
@printf("Transformed series: μ=%.3f  σ=%.3f  min=%.3f  max=%.3f\n",
        mean(y), std(y), minimum(y), maximum(y))

# ── Parameters ─────────────────────────────────────────────────────────────────

const Q      = 27     # one solar rotation (Carrington period ≈ 27 days)
const W      = 365    # topo window (1 year)
const K      = 20     # refit cadence
const N_TEST = 1000   # test points (≈ 2.7 years)
const H_LIST = [1, 27]

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
println("\nEmbedding: τ=$τ, dim=$d  →  $p_topo topo features")

# ── Topo feature matrix ────────────────────────────────────────────────────────

n_windows  = N - W + 1
ckpt_topo  = "checkpoints/sunspots/F_topo.jls"

if isfile(ckpt_topo)
    println("Loading topo feature matrix from checkpoint...")
    F_topo, tgrid_land = deserialize(ckpt_topo)
    println("Loaded: $(size(F_topo, 1)) × $(size(F_topo, 2))")
else
    println("Computing topo features ($n_windows windows)...")
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
        i % 2000 == 1 && @printf("  window %d/%d\n", i, n_windows)
        F_topo[i, :] = topo_features(y[i:i+W-1]; spec=spec,
                                      tgrid_landscape=tgrid_land,
                                      tgrid_betti=nothing)
    end
    F_topo[isnan.(F_topo) .| isinf.(F_topo)] .= 0.0
    serialize(ckpt_topo, (F_topo, tgrid_land))
    println("Done. Checkpointed.")
end

# ── Walk-forward ───────────────────────────────────────────────────────────────

σ = std(y)
results = Dict{Int, NamedTuple}()

for h in H_LIST
    n_valid    = n_windows - h
    test_start = max(n_valid - N_TEST + 1, Q + 1)
    n_test_act = n_valid - test_start + 1

    println("\n", "="^55)
    println("Horizon h=$h  |  test points: $n_test_act")
    println("="^55)

    ae_base = Float64[]; ae_aug = Float64[]
    bst_b   = nothing;   bst_a  = nothing
    make_lag(i) = y[i+W-Q : i+W-1]

    for t_test in test_start:n_valid
        t_train = t_test - 1

        if isnothing(bst_b) || (t_test - test_start) % K == 0
            X_lag_tr = reduce(vcat, make_lag(i)' for i in 1:t_train)
            X_top_tr = F_topo[1:t_train, :]
            y_tr     = [y[i+W-1+h] for i in 1:t_train]
            X_aug_tr = hcat(X_lag_tr, X_top_tr)
            bst_b    = xgboost((X_lag_tr, y_tr); xgb_params...)
            bst_a    = xgboost((X_aug_tr, y_tr); xgb_params...)
        end

        X_lag_te = make_lag(t_test)'
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
            Δmae > 2 ? "TOPO HELPS ✓" : Δmae < -2 ? "TOPO HURTS" : "NEUTRAL")
end

# ── Save results ───────────────────────────────────────────────────────────────

serialize("checkpoints/sunspots/results.jls", results)

csv_rows = [(h=h, mae_baseline=results[h].mae_b, mae_augmented=results[h].mae_a,
             delta_mae_pct=results[h].Δmae, rmse_baseline=results[h].rmse_b,
             rmse_augmented=results[h].rmse_a, delta_rmse_pct=results[h].Δrmse,
             n_test=results[h].n) for h in H_LIST]
CSV.write("results/sunspot_forecasting.csv", DataFrame(csv_rows))

# ── Summary ────────────────────────────────────────────────────────────────────

println("\n", "="^55)
println("FINAL SUMMARY  (normalised by σ=$(round(σ,digits=4)))")
println("Series: SILSO daily sunspot number, sqrt(SN+1) transform")
println("Embedding: τ=$τ, dim=$d  |  Window W=$W  |  Q=$Q lags")
println("="^55)
@printf("%-4s  %-12s  %7s  %7s  %7s  %7s\n",
        "h", "Model", "MAE", "RMSE", "ΔMAE%", "ΔRMSE%")
println("-"^55)
for h in H_LIST
    r = results[h]
    @printf("%-4d  %-12s  %7.4f  %7.4f\n",            h, "Baseline",  r.mae_b, r.rmse_b)
    @printf("%-4s  %-12s  %7.4f  %7.4f  %+6.2f%%  %+6.2f%%\n",
            "", "Augmented", r.mae_a, r.rmse_a, r.Δmae, r.Δrmse)
    println("-"^55)
end
println("Results saved to results/sunspot_forecasting.csv")
