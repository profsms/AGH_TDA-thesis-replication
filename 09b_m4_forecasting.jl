# 09b_m4_forecasting.jl
# M4 daily series: baseline lags vs. baseline + topo features
# Evaluation: sMAPE and MASE (M4 standard metrics)
# Horizon: h=1 and h=5 (M4 daily horizon is 1..14; we report 1 and 5)
# CV: expanding walk-forward, refit every K=20 steps, test on last 500 obs
# Transformation: first differences of log(value) — stationary, matches financial panel

using TopoTS
using XGBoost
using CSV
using DataFrames
using Statistics
using Printf
using Glob
using Random
using Serialization

Random.seed!(42)

# ── Parameters ─────────────────────────────────────────────────────────────────

const Q         = 5
const W         = 252
const K         = 20
const N_TEST    = 500
const H_LIST    = [1, 5]

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
p_topo = length(feature_names(base_spec))
xgb_params = (num_round=100, max_depth=4, eta=0.05,
               subsample=0.8, objective="reg:squarederror", verbosity=0)

# ── M4 evaluation metrics ──────────────────────────────────────────────────────

function smape(y_true, y_pred)
    n = length(y_true)
    100 / n * sum(2abs.(y_true .- y_pred) ./ (abs.(y_true) .+ abs.(y_pred)))
end

function mase(y_true, y_pred, y_train; m=1)
    # m=1 for non-seasonal (daily M4 uses m=1 for MASE denominator)
    mae_forecast = mean(abs.(y_true .- y_pred))
    n = length(y_train)
    mae_naive = mean(abs.(y_train[m+1:end] .- y_train[1:end-m]))
    return mae_forecast / mae_naive
end

# ── Series loop ────────────────────────────────────────────────────────────────

mkpath("checkpoints/m4")
mkpath("results")

series_files = sort(glob("data/m4/D*.csv"))
# Exclude the meta CSVs
series_files = filter(f -> occursin(r"D\d+\.csv", f), series_files)

println("Found $(length(series_files)) M4 daily series")
println("="^70)

# Aggregate storage: smape and mase per horizon
agg = Dict(h => (smape_b=Float64[], smape_a=Float64[],
                  mase_b=Float64[], mase_a=Float64[]) for h in H_LIST)
series_results = []   # for per-series table

for fpath in series_files
    sid  = replace(basename(fpath), ".csv" => "")

    # Skip if already checkpointed
    ckpt_path = "checkpoints/m4/$(sid).jls"
    if isfile(ckpt_path)
        println("SKIP $sid (checkpoint exists)")
        agg_data = deserialize(ckpt_path)
        for h in H_LIST
            haskey(agg_data, h) || continue
            r = agg_data[h]
            push!(agg[h].smape_b, r.smape_b); push!(agg[h].smape_a, r.smape_a)
            push!(agg[h].mase_b,  r.mase_b);  push!(agg[h].mase_a,  r.mase_a)
            push!(series_results, merge((sid=sid,), (
                Symbol("smape_b_h$h") => round(r.smape_b, digits=3),
                Symbol("smape_a_h$h") => round(r.smape_a, digits=3),
                Symbol("mase_b_h$h")  => round(r.mase_b,  digits=3),
                Symbol("mase_a_h$h")  => round(r.mase_a,  digits=3),
                Symbol("delta_h$h")   => round((r.smape_b - r.smape_a)/r.smape_b*100, digits=2),
            )))
        end
        continue
    end

    df   = CSV.read(fpath, DataFrame)
    sort!(df, :t)

    # Work on log-differences of the level series
    levels  = df.value
    returns = diff(log.(levels))   # log-returns
    N       = length(returns)
    σ       = std(returns)

    if N < W + N_TEST + maximum(H_LIST) + Q
        println("SKIP $sid — N=$N too short")
        continue
    end

    # Per-series embedding parameters
    τ = optimal_lag(returns)
    d = optimal_dim(returns; lag=τ, max_dim=5)
    d = clamp(d, 2, 5)

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

    print("$sid (N=$N, τ=$τ, dim=$d)  ")

    # Topo feature matrix
    n_windows = N - W + 1
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
        F_topo[i, :] = topo_features(returns[i:i+W-1]; spec=spec,
                                      tgrid_landscape=tgrid_land,
                                      tgrid_betti=nothing)
    end
    F_topo[isnan.(F_topo) .| isinf.(F_topo)] .= 0.0

    row_result = (sid=sid,)

    for h in H_LIST
        n_valid    = n_windows - h
        test_start = max(n_valid - N_TEST + 1, Q + 1)

        ae_base = Float64[]; ae_aug = Float64[]
        # For sMAPE/MASE we need predictions in level space
        pred_base_level = Float64[]; pred_aug_level = Float64[]
        true_level      = Float64[]

        bst_b = nothing; bst_a = nothing
        make_lag(i) = returns[i+W-Q : i+W-1]

        for t_test in test_start:n_valid
            t_train = t_test - 1

            if isnothing(bst_b) || (t_test - test_start) % K == 0
                X_lag_tr = reduce(vcat, make_lag(i)' for i in 1:t_train)
                X_top_tr = F_topo[1:t_train, :]
                y_tr     = [returns[i+W-1+h] for i in 1:t_train]
                X_aug_tr = hcat(X_lag_tr, X_top_tr)
                bst_b = xgboost((X_lag_tr, y_tr); xgb_params...)
                bst_a = xgboost((X_aug_tr, y_tr); xgb_params...)
            end

            X_lag_te = make_lag(t_test)'
            X_aug_te = hcat(X_lag_te, F_topo[t_test:t_test, :])
            y_te     = returns[t_test+W-1+h]

            # Predicted log-return → predicted level
            ep       = t_test + W - 1        # endpoint index in returns
            lv_ep    = levels[ep + 1]        # level at endpoint (returns[ep] = log(levels[ep+1]/levels[ep]))
            pred_r_b = XGBoost.predict(bst_b, X_lag_te)[1]
            pred_r_a = XGBoost.predict(bst_a, X_aug_te)[1]

            push!(pred_base_level, lv_ep * exp(pred_r_b))
            push!(pred_aug_level,  lv_ep * exp(pred_r_a))
            push!(true_level,      levels[ep + 1 + h])

            push!(ae_base, abs(pred_r_b - y_te))
            push!(ae_aug,  abs(pred_r_a - y_te))
        end

        # Training returns for MASE denominator
        y_train_returns = returns[1 : test_start + W - 2]

        sm_b = smape(true_level, pred_base_level)
        sm_a = smape(true_level, pred_aug_level)
        ms_b = mase(true_level, pred_base_level, levels[1:test_start+W]; m=1)
        ms_a = mase(true_level, pred_aug_level,  levels[1:test_start+W]; m=1)

        push!(agg[h].smape_b, sm_b); push!(agg[h].smape_a, sm_a)
        push!(agg[h].mase_b,  ms_b); push!(agg[h].mase_a,  ms_a)

        Δsmape = (sm_b - sm_a) / sm_b * 100
        @printf("h=%d: sMAPE %.2f→%.2f (%+.1f%%)  ", h, sm_b, sm_a, Δsmape)

        row_result = merge(row_result, (
            Symbol("smape_b_h$h") => round(sm_b, digits=3),
            Symbol("smape_a_h$h") => round(sm_a, digits=3),
            Symbol("mase_b_h$h")  => round(ms_b, digits=3),
            Symbol("mase_a_h$h")  => round(ms_a, digits=3),
            Symbol("delta_h$h")   => round(Δsmape, digits=2),
        ))
    end
    println()
    push!(series_results, row_result)

    # Checkpoint this series
    series_data = Dict(h => (
        smape_b = agg[h].smape_b[end],
        smape_a = agg[h].smape_a[end],
        mase_b  = agg[h].mase_b[end],
        mase_a  = agg[h].mase_a[end],
    ) for h in H_LIST)
    serialize(ckpt_path, series_data)
end

# ── Summary ────────────────────────────────────────────────────────────────────

println("\n", "="^80)
println("M4 DAILY — AGGREGATE RESULTS (mean across series)")
println("="^80)
@printf("%-6s  %10s  %10s  %8s  %10s  %10s  %8s\n",
        "h", "sMAPE-B", "sMAPE-A", "ΔsMAPE%", "MASE-B", "MASE-A", "ΔMASE%")
println("-"^80)
for h in H_LIST
    a = agg[h]
    sm_b = mean(a.smape_b); sm_a = mean(a.smape_a)
    ms_b = mean(a.mase_b);  ms_a = mean(a.mase_a)
    Δsm = (sm_b - sm_a) / sm_b * 100
    Δms = (ms_b - ms_a) / ms_b * 100
    verdict = Δsm > 2 ? "↑ TOPO HELPS" : Δsm < -2 ? "↓ TOPO HURTS" : "= NEUTRAL"
    @printf("%-6d  %10.3f  %10.3f  %+7.2f%%  %10.3f  %10.3f  %+7.2f%%  %s\n",
            h, sm_b, sm_a, Δsm, ms_b, ms_a, Δms, verdict)
end
println("="^80)

# Per-series table
println("\nPER-SERIES RESULTS (sMAPE, baseline → augmented, Δ%)")
println("="^80)
@printf("%-8s", "ID")
for h in H_LIST
    @printf("  %10s  %10s  %7s", "sMAPE-B h=$h", "sMAPE-A h=$h", "Δ h=$h")
end
println()
println("-"^80)
for r in series_results
    @printf("%-8s", r.sid)
    for h in H_LIST
        @printf("  %10.3f  %10.3f  %+6.2f%%",
                r[Symbol("smape_b_h$h")],
                r[Symbol("smape_a_h$h")],
                r[Symbol("delta_h$h")])
    end
    println()
end
println("="^80)
