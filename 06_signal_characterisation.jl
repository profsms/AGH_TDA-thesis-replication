# =============================================================================
# 06_signal_characterisation.jl — §6.1 Signal Characterisation
#
# Demonstrates the TopoTS.jl pipeline on synthetic signals with known
# dynamical structure. For each signal type we show:
#   1. Time series plot
#   2. Delay embedding (3D projection)
#   3. H₁ persistence diagram
#   4. Betti curve β₁(ε)
#   5. CROCKER plot (evolution across sliding windows)
#
# Signal types:
#   A. Pure periodic   — sinusoid at 10 Hz → clean circle in phase space
#   B. Quasiperiodic   — sum of 10 Hz + √2·7 Hz → torus in phase space
#   C. Chaotic (Lorenz) — strange attractor → complex H₁ structure
#   D. AR(1) noise     — no periodic structure → no H₁ features
#
# Run: julia --project=. 06_signal_characterisation.jl
# =============================================================================

using TopoTS
using CairoMakie
using Printf
using DifferentialEquations
using Random, Statistics, LinearAlgebra

mkpath("plots/signal_characterisation")

const FS = 200.0   # sampling rate (Hz)
const T  = 2000    # samples per signal (~10 s)

Random.seed!(42)

# =============================================================================
# 1. Signal generators
# =============================================================================

# A. Pure periodic: sinusoid with additive noise
function gen_periodic(; T=T, f=10.0, snr=10.0)
    t   = (0:T-1) ./ FS
    sig = sin.(2π .* f .* t)
    sig .+ randn(T) ./ snr
end

# B. Quasiperiodic: incommensurable frequencies (ratio √2)
function gen_quasiperiodic(; T=T, f1=10.0, snr=10.0)
    t   = (0:T-1) ./ FS
    f2  = f1 * sqrt(2)
    sig = sin.(2π .* f1 .* t) .+ 0.7 .* sin.(2π .* f2 .* t)
    sig .+ randn(T) ./ snr
end

# C. Lorenz attractor (x-coordinate, locally stationary segment)
function gen_lorenz(; T=T, sigma=10.0, rho=28.0, beta=8/3,
                     dt=0.01, warmup=5000)
    function lorenz!(du, u, p, t)
        σ, ρ, β = p
        du[1] = σ*(u[2]-u[1])
        du[2] = u[1]*(ρ-u[3])-u[2]
        du[3] = u[1]*u[2]-β*u[3]
    end
    u0  = [1.0, 0.0, 0.0]
    tspan = (0.0, (T + warmup) * dt)
    prob  = ODEProblem(lorenz!, u0, tspan, [sigma, rho, beta])
    sol   = solve(prob, Tsit5(); saveat=dt, abstol=1e-8, reltol=1e-8)
    x = [sol[1, i] for i in warmup+1:warmup+T]
    # Normalise to unit variance
    x ./= std(x)
end

# D. AR(1) noise — no periodic structure
function gen_ar1(; T=T, phi=0.90)
    x = zeros(T); x[1] = randn()
    for t in 2:T; x[t] = phi*x[t-1] + randn()*sqrt(1-phi^2); end
    return x
end

SIGNALS = [
    ("periodic",      "A: Pure periodic (10 Hz)",    gen_periodic()),
    ("quasiperiodic", "B: Quasiperiodic (10 + sqrt(2)*7 Hz)", gen_quasiperiodic()),
    ("lorenz",        "C: Lorenz attractor (x-coord)", gen_lorenz()),
    ("ar1",           "D: AR(1) noise (phi=0.90)",  gen_ar1()),
]

to_pairs(dgm) = [(Float64(b), Float64(d)) for (b,d) in dgm if isfinite(d)]

# =============================================================================
# 2. Compute embeddings and diagrams for all signals
# =============================================================================

println("Computing embeddings and persistence diagrams...")

RESULTS = map(SIGNALS) do (key, label, sig)
    tau  = optimal_lag(sig)
    emb  = embed(sig; dim=3, lag=tau)
    ph   = persistent_homology(emb; dim_max=1)
    h0   = to_pairs(ph[1])
    h1   = to_pairs(ph[2])
    @printf("  %-16s  τ*=%2d  |H₀|=%3d  |H₁|=%3d  max H₁ pers=%.4f\n",
            key, tau, length(h0), length(h1),
            isempty(h1) ? 0.0 : maximum(d-b for (b,d) in h1))
    (key=key, label=label, sig=sig, tau=tau, emb=emb, h0=h0, h1=h1)
end

# =============================================================================
# 3. Figure 1: Time series overview (2×2 grid)
# =============================================================================

fig_ts = Figure(size=(1200, 800))
t_axis = (0:T-1) ./ FS

for (i, r) in enumerate(RESULTS)
    row = div(i-1, 2) + 1; col = mod(i-1, 2) + 1
    ax  = Axis(fig_ts[row, col];
               title  = r.label,
               xlabel = "Time (s)", ylabel = "Amplitude",
               titlesize = 12)
    lines!(ax, t_axis[1:500], r.sig[1:500]; color=:royalblue, linewidth=0.8)
end

save("plots/signal_characterisation/characterisation_timeseries.pdf", fig_ts)
println("\nSaved → plots/signal_characterisation/characterisation_timeseries.pdf")

# =============================================================================
# 4. Figure 2: Persistence diagrams (2×2, H₁ only)
# =============================================================================

fig_pd = Figure(size=(1100, 1000))

for (i, r) in enumerate(RESULTS)
    row = div(i-1, 2) + 1; col = mod(i-1, 2) + 1
    ax  = Axis(fig_pd[row, col];
               title  = r.label,
               xlabel = "Birth", ylabel = "Death",
               aspect = 1, titlesize = 12)

    # Diagonal
    all_vals = vcat([b for (b,_) in r.h1], [d for (_,d) in r.h1],
                    [b for (b,_) in r.h0], [d for (_,d) in r.h0])
    isempty(all_vals) && (all_vals = [0.0, 1.0])
    lo = minimum(all_vals); hi = maximum(all_vals)
    lines!(ax, [lo, hi], [lo, hi]; color=:gray70, linewidth=1)

    # H₀ points (faint)
    if !isempty(r.h0)
        scatter!(ax, [b for (b,_) in r.h0], [d for (_,d) in r.h0];
                 color=(:steelblue, 0.3), markersize=5, label="H₀")
    end

    # H₁ points (prominent)
    if !isempty(r.h1)
        scatter!(ax, [b for (b,_) in r.h1], [d for (_,d) in r.h1];
                 color=(:firebrick, 0.8), markersize=7, label="H₁")
        # Highlight most persistent H₁ point
        best = argmax([d-b for (b,d) in r.h1])
        b, d = r.h1[best]
        scatter!(ax, [b], [d]; color=:gold, markersize=12,
                 marker=:star5, label="max H₁")
    end

    (row==1 && col==1) && axislegend(ax; position=:rb, framevisible=false,
                                      labelsize=9)
end

save("plots/signal_characterisation/characterisation_diagrams.pdf", fig_pd)
println("Saved → plots/signal_characterisation/characterisation_diagrams.pdf")

# =============================================================================
# 5. Figure 3: Betti curves β₁(ε) for all four signals
# =============================================================================

function betti_curve(pairs, eps_grid)
    [count(((b,d),) -> b ≤ ε ≤ d, pairs) for ε in eps_grid]
end

fig_bc = Figure(size=(1100, 400))
ax_bc  = Axis(fig_bc[1, 1];
              title  = "H1 Betti curves by signal type",
              xlabel = "Filtration radius ε",
              ylabel = "β₁(ε)")

COLORS = [:royalblue, :darkorange, :firebrick, :gray50]
STYLES = [:solid, :solid, :solid, :dash]

# Global ε range
all_h1 = vcat([r.h1 for r in RESULTS]...)
if !isempty(all_h1)
    eps_lo = quantile([b for (b,_) in all_h1], 0.02)
    eps_hi = quantile([d for (_,d) in all_h1], 0.98)
    eps_grid = range(eps_lo, eps_hi; length=200)

    for (r, col, ls) in zip(RESULTS, COLORS, STYLES)
        bc = betti_curve(r.h1, eps_grid)
        lines!(ax_bc, collect(eps_grid), Float64.(bc);
               color=col, linestyle=ls, linewidth=2,
               label=r.label)
    end
    axislegend(ax_bc; position=:rt, framevisible=false, labelsize=9)
end

save("plots/signal_characterisation/characterisation_betti.pdf", fig_bc)
println("Saved → plots/signal_characterisation/characterisation_betti.pdf")

# =============================================================================
# 6. Figure 4: CROCKER plots — sliding window evolution
#    Use a shorter signal with explicit window stride to show dynamics
# =============================================================================

println("\nComputing CROCKER plots (sliding window)...")

WIN   = 300   # window length (samples) — ~1.5 s
STEP  = 20    # stride

function sliding_h1(sig; win=WIN, step=STEP, dim=3)
    N      = length(sig)
    starts = 1:step:N-win+1
    tau    = optimal_lag(sig)
    map(starts) do s
        seg = sig[s:s+win-1]
        emb = embed(seg; dim=dim, lag=tau)
        ph  = persistent_homology(emb; dim_max=1)
        to_pairs(ph[2])
    end
end

function crocker_from_windows(windows; n_eps=40)
    all_b = [b for w in windows for (b,_) in w]
    all_d = [d for w in windows for (_,d) in w]
    isempty(all_b) && return zeros(Int, n_eps, length(windows)),
                              range(0, 1; length=n_eps)
    eps_grid = range(quantile(all_b, 0.05), quantile(all_d, 0.95);
                     length=n_eps)
    C = zeros(Int, n_eps, length(windows))
    for (j, w) in enumerate(windows)
        for (k, ε) in enumerate(eps_grid)
            C[k, j] = count(((b,d),) -> b ≤ ε ≤ d, w)
        end
    end
    return C, eps_grid
end

# Only compute CROCKER for periodic and Lorenz (most informative)
CROCKER_SIGNALS = [
    ("periodic",  "A: Pure periodic",      RESULTS[1].sig),
    ("lorenz",    "C: Lorenz attractor",    RESULTS[3].sig),
]

fig_cr = Figure(size=(1200, 700))

for (col, (key, label, sig)) in enumerate(CROCKER_SIGNALS)
    print("  $label ... "); flush(stdout)
    wins = sliding_h1(sig)
    C, eps = crocker_from_windows(wins)
    println("$(size(C, 2)) windows, β₁ max = $(maximum(C))")

    ax = Axis(fig_cr[1, col];
              title  = label,
              xlabel = "Window index",
              ylabel = "Filtration radius ε",
              titlesize = 12)
    hm = heatmap!(ax, 1:size(C,2), collect(eps), C';
                  colormap=Reverse(:deep), interpolate=false)
    col == 2 && Colorbar(fig_cr[1, 3], hm; label="β₁(ε, window)")
end

save("plots/signal_characterisation/characterisation_crocker.pdf", fig_cr)
println("Saved → plots/signal_characterisation/characterisation_crocker.pdf")

# =============================================================================
# 7. Summary statistics
# =============================================================================

println("\n=== Summary ===")
@printf("%-16s  %4s  %6s  %6s  %10s\n",
        "Signal", "τ*", "|H₁|", "maxH₁", "interpretation")
println("-"^60)
for r in RESULTS
    max_h1 = isempty(r.h1) ? 0.0 : maximum(d-b for (b,d) in r.h1)
    interp = if r.key == "periodic";      "clean loop (S¹)"
             elseif r.key == "quasiperiodic"; "torus structure (T²)"
             elseif r.key == "lorenz";    "strange attractor"
             else;                         "no loop structure"
             end
    @printf("%-16s  %4d  %6d  %6.4f  %s\n",
            r.key, r.tau, length(r.h1), max_h1, interp)
end

println("\nAll figures saved to plots/signal_characterisation/")
