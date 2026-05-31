# =============================================================================
# 01b_synthetic_phasespace.jl — Phase-space pipeline benchmark (T=1024)
#
# Companion to 01_synthetic.jl. Longer trials give Pipeline B sufficient
# point-cloud density for reliable H₁ detection.
# All 3 metrics in ONE pass. Parallelised over replicates. Checkpointed.
#
# Run: julia --threads=auto --project=. 01b_synthetic_phasespace.jl
# =============================================================================

using TopoTS, FFTW
using CairoMakie
using Random, Statistics, StatsBase, Printf
using Serialization

mkpath("plots/empirical")
mkpath("checkpoints")

const FS       = 100.0
const T_TRIAL  = 1024
const R_TRIALS = 200
const R_STAR   = 100
const R0       = 20

# =============================================================================
# Checkpoint helpers
# =============================================================================

ckpt_path(key) = joinpath("checkpoints", key * ".jls")

function cached(fn::Function, key::String)
    p = ckpt_path(key)
    if isfile(p)
        println("  [ckpt] loading $key ...")
        return deserialize(p)
    end
    r = fn()
    serialize(p, r)
    println("  [ckpt] saved $key")
    return r
end

# =============================================================================
# 1. Signal generators
# =============================================================================

function ar2(T::Int, f::Float64;
             r=0.95, sigma=0.1, rng=Random.default_rng())
    φ1 = 2r * cos(2π * f / FS); φ2 = -r^2
    x  = zeros(T)
    x[1] = sigma*randn(rng); x[2] = sigma*randn(rng)
    for t in 3:T
        x[t] = φ1*x[t-1] + φ2*x[t-2] + sigma*randn(rng)
    end
    return x
end

function ar1(T::Int; phi=0.90, sigma=0.1, rng=Random.default_rng())
    x = zeros(T); x[1] = sigma*randn(rng)
    for t in 2:T; x[t] = phi*x[t-1] + sigma*randn(rng); end
    return x
end

function gen_s1(; R=R_TRIALS, T=T_TRIAL, rstar=R_STAR, seed=1)
    [r <= rstar ? ar2(T, 15.0; rng=MersenneTwister(seed*1000+r)) :
                  ar1(T;       rng=MersenneTwister(seed*1000+r))
     for r in 1:R]
end
gen_null_s1(; R=R_TRIALS, T=T_TRIAL, seed=1) =
    [ar2(T, 15.0; rng=MersenneTwister(seed*1000+r)) for r in 1:R]

function gen_s2(; R=R_TRIALS, T=T_TRIAL, rstar=R_STAR, seed=1)
    map(1:R) do r
        rng = MersenneTwister(seed*1000+r)
        r <= rstar ? ar2(T, 10.0; rng) :
            (s = ar2(T, 10.0; rng) .+ ar2(T, 37.0; rng); s ./= std(s))
    end
end
gen_null_s2(; R=R_TRIALS, T=T_TRIAL, seed=1) =
    [ar2(T, 10.0; rng=MersenneTwister(seed*1000+r)) for r in 1:R]

function gen_s3(; R=R_TRIALS, T=T_TRIAL, rstar=R_STAR, seed=1)
    [ar2(T, r <= rstar ? 10.0 : 40.0; rng=MersenneTwister(seed*1000+r))
     for r in 1:R]
end
gen_null_s3(; R=R_TRIALS, T=T_TRIAL, seed=1) =
    [ar2(T, 10.0; rng=MersenneTwister(seed*1000+r)) for r in 1:R]

# =============================================================================
# 2. Core pipeline — L1, T, B in ONE pass
# =============================================================================

function smoothed_periodogram(sig; bw=5)
    P = abs2.(rfft(sig)) ./ length(sig)
    return [mean(P[max(1,i-bw):min(length(P),i+bw)]) for i in 1:length(P)]
end

to_pairs(dgm) = [(Float64(b), Float64(d)) for (b,d) in dgm if isfinite(d)]

function h1_pairs(ph)
    try
        length(ph) >= 2 ? to_pairs(ph[2]) : Tuple{Float64,Float64}[]
    catch
        Tuple{Float64,Float64}[]
    end
end

function all_metrics(trials::Vector{Vector{Float64}}; bw=5, r0=R0)
    R     = length(trials)
    Ps    = [smoothed_periodogram(t; bw) for t in trials]
    D_L1  = zeros(R-1)
    D_T   = zeros(R-1)
    S_sum = copy(Ps[1])
    for r in 1:R-1
        P_avg   = S_sum ./ r
        D_L1[r] = sum(abs, P_avg .- Ps[r+1])
        d1 = sublevel_ph(P_avg).H0
        d2 = sublevel_ph(Ps[r+1]).H0
        D_T[r]  = wasserstein_distance(d1, d2)
        S_sum  .+= Ps[r+1]
    end
    ref  = reduce(vcat, trials[1:r0])
    tau  = optimal_lag(ref)
    dgms = [h1_pairs(persistent_homology(
                embed(t; dim=3, lag=tau); dim_max=1))
            for t in trials]
    D_B  = [wasserstein_distance(dgms[r], dgms[r+1]) for r in 1:R-1]
    return (L1=D_L1, T=D_T, B=D_B)
end

# =============================================================================
# 3. CUSUM
# =============================================================================

cusum(D, mu) = accumulate((c, d) -> max(0.0, c + d - mu), D; init=0.0)

function normalise(D, r0)
    mu = mean(D[1:r0]); sg = std(D[1:r0])
    return (D .- mu) ./ max(sg, 1e-12)
end

# =============================================================================
# 4. MC evaluation with progress and checkpoints
# =============================================================================

const METRICS = (:L1, :T, :B)

function mc_eval_all(gen, gnull, scenario_key;
                     B=500, B0=200, alpha=0.01, r0=R0, rstar=R_STAR)

    τ = cached("synB_$(scenario_key)_tau") do
        null_maxC = Dict(m => zeros(B0) for m in METRICS)
        done = Threads.Atomic{Int}(0)
        lk   = ReentrantLock()
        println("    calibrating (B₀=$B0 null replicates):")
        Threads.@threads for b in 1:B0
            scores = all_metrics(gnull(; seed=b + 10_000))
            for m in METRICS
                D  = getfield(scores, m)
                Dn = normalise(D, min(r0, length(D)))
                lock(lk) do
                    null_maxC[m][b] = maximum(cusum(Dn, 0.0))
                end
            end
            n = Threads.atomic_add!(done, 1) + 1
            print("\r      replicate $n/$B0"); flush(stdout)
        end
        println()
        Dict(m => quantile(null_maxC[m], 1.0 - alpha) for m in METRICS)
    end

    results = cached("synB_$(scenario_key)_results") do
        delays = Dict(m => fill(typemax(Int), B) for m in METRICS)
        missed = Dict(m => zeros(Bool, B)        for m in METRICS)
        done   = Threads.Atomic{Int}(0)
        println("    evaluating (B=$B replicates):")
        Threads.@threads for b in 1:B
            scores = all_metrics(gen(; seed=b))
            for m in METRICS
                D  = getfield(scores, m)
                length(D) < r0 + 2 && (missed[m][b]=true; continue)
                Dn = normalise(D, r0)
                C  = cusum(Dn, 0.0)
                rh = findfirst(>(τ[m]), @view C[r0+1:end])
                isnothing(rh) ? (missed[m][b]=true) :
                                (delays[m][b] = rh - (rstar - r0))
            end
            n = Threads.atomic_add!(done, 1) + 1
            print("\r      replicate $n/$B"); flush(stdout)
        end
        println()
        Dict{Symbol,NamedTuple}(m => begin
            valid = delays[m][.!missed[m]]
            (med  = isempty(valid) ? NaN : median(valid),
             iqr  = isempty(valid) ? NaN : iqr(valid),
             miss = sum(missed[m]) / B,
             tau  = τ[m])
        end for m in METRICS)
    end

    return results
end

# =============================================================================
# 5. Run
# =============================================================================

SCENARIOS = [
    ("s1", "1: orbit collapse (AR2→AR1)",  gen_s1, gen_null_s1,
     "Loop collapses; B advantage expected"),
    ("s2", "2: quasiperiodic onset",        gen_s2, gen_null_s2,
     "New H₁ generator; both detect"),
    ("s3", "3: frequency shift (sanity)",   gen_s3, gen_null_s3,
     "Topology unchanged; A dominates"),
]

println("Threads: $(Threads.nthreads())")
println("=== Phase-space benchmark  R=$R_TRIALS × T=$T_TRIAL, r*=$R_STAR ===\n")

RES = Dict{String, Any}()

for (key, name, gen, gnull, desc) in SCENARIOS
    println("Scenario $name")
    println("  [$desc]")
    res = mc_eval_all(gen, gnull, key; B=500, B0=200)
    RES[name] = res
    for (sym, label) in [(:L1,"A: L1 spectral"),
                          (:T, "A: T topo H₀"),
                          (:B, "B: T-PS phase H₁")]
        r = res[sym]
        @printf("  %-22s  med Δ=%6.1f  IQR=%6.1f  miss=%5.1f%%\n",
                label, r.med, r.iqr, r.miss*100)
    end
    println()
end

# =============================================================================
# 6. Figure
# =============================================================================

fig = Figure(size=(1300, 1050))

for (row, (_, name, gen, _, desc)) in enumerate(SCENARIOS)
    scores = all_metrics(gen(; seed=1))
    DN_L1  = normalise(scores.L1, R0)
    DN_T   = normalise(scores.T,  R0)
    DN_B   = normalise(scores.B,  R0)
    τ_L1   = RES[name][:L1].tau
    τ_T    = RES[name][:T].tau
    τ_B    = RES[name][:B].tau

    ax1 = Axis(fig[row, 1];
               title     = "Scenario $name — distances\n($desc)",
               xlabel    = "Trial r", ylabel = "Normalised distance",
               titlesize = 10)
    lines!(ax1, DN_L1; color=:steelblue,  label="A: L1 spectral")
    lines!(ax1, DN_T;  color=:firebrick,  label="A: T topo H₀")
    lines!(ax1, DN_B;  color=:darkorange, linewidth=2,
           label="B: T-PS phase-space H₁")
    vlines!(ax1, [R_STAR]; color=:black, linestyle=:dash, label="True r*")
    row == 1 && axislegend(ax1; position=:rt, framevisible=false, labelsize=9)

    ax2 = Axis(fig[row, 2];
               title  = "Scenario $name — CUSUM",
               xlabel = "Trial r", ylabel = "Cᵣ", titlesize=10)
    lines!(ax2, cusum(DN_L1, 0.0); color=:steelblue)
    lines!(ax2, cusum(DN_T,  0.0); color=:firebrick)
    lines!(ax2, cusum(DN_B,  0.0); color=:darkorange, linewidth=2)
    hlines!(ax2, [τ_L1]; color=:steelblue,  linestyle=:dot)
    hlines!(ax2, [τ_T];  color=:firebrick,  linestyle=:dot)
    hlines!(ax2, [τ_B];  color=:darkorange, linestyle=:dot)
    vlines!(ax2, [R_STAR]; color=:black, linestyle=:dash)
end

save("plots/empirical/synthetic_phasespace.pdf", fig)
println("Saved → plots/empirical/synthetic_phasespace.pdf")

# =============================================================================
# 7. LaTeX table
# =============================================================================

println("\n=== LaTeX rows (T=$T_TRIAL) ===")
@printf("%-32s & A-L1 miss & A-T miss & B miss & A-L1 Δ & A-T Δ & B Δ \\\\\n",
        "Scenario")
println("\\midrule")
for (_, name, _, _, _) in SCENARIOS
    res = RES[name]
    @printf("%-32s & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f \\\\\n",
            name,
            res[:L1].miss*100, res[:T].miss*100, res[:B].miss*100,
            res[:L1].med,      res[:T].med,      res[:B].med)
end
