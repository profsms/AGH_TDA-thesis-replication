# 07_period_estimation.jl
# Simulation: topological vs FFT period estimator
# fs=500 Hz, T=2000 samples, periods=[50,100,200], SNR=[5,10,20] dB
# Waveforms: sinusoid, sawtooth, square — B=100 Monte Carlo replicates

using TopoTS
using FFTW
using Statistics
using Printf
using Random

Random.seed!(42)

# ── Signal generators ──────────────────────────────────────────────────────────

make_sinusoid(P, T) = sin.(2π .* (0:T-1) ./ P)
make_sawtooth(P, T) = mod.((0:T-1) ./ P, 1.0) .* 2 .- 1
make_square(P, T)   = sign.(sin.(2π .* (0:T-1) ./ P))

function add_noise(signal, snr_db)
    noise_power = mean(signal .^ 2) / (10^(snr_db / 10))
    signal .+ randn(length(signal)) .* sqrt(noise_power)
end

# ── Max H₁ persistence from a DiagramCollection ───────────────────────────────

function max_h1_persistence(dc::DiagramCollection)
    length(dc) < 2 && return 0.0          # no H₁ computed
    h1 = dc[2]
    isempty(h1) && return 0.0
    return maximum((p[2] - p[1] for p in h1 if isfinite(p[2])), init=0.0)
end

# ── Topological period estimator ───────────────────────────────────────────────
# Scan candidate window lengths M. For each M, run windowed_ph with step=M
# (non-overlapping windows of exactly length M) and record the max H₁
# persistence across all windows. Return the M that maximises it.

function topo_period_estimate(x, M_grid; dim=3)
    τ = optimal_lag(x)
    min_window = (dim - 1) * τ + 1
    valid_M = filter(M -> M >= min_window, M_grid)
    isempty(valid_M) && return first(M_grid)

    best_M    = first(valid_M)
    best_pers = -Inf
    for M in valid_M
        wd   = windowed_ph(x; window=M, step=M, dim=dim, lag=τ)
        pers = maximum(max_h1_persistence(dc) for dc in wd.diagrams; init=0.0)
        if pers > best_pers
            best_pers = pers
            best_M    = M
        end
    end
    return best_M
end

# ── FFT period estimator ───────────────────────────────────────────────────────

function fft_period_estimate(x, fs)
    N     = length(x)
    psd   = abs2.(rfft(x))[2:end]
    freqs = (1:length(psd)) ./ N .* fs
    return fs / freqs[argmax(psd)]
end

# ── Parameters ─────────────────────────────────────────────────────────────────

const fs      = 500
const T       = 2000
const periods = [50, 100, 200]
const snrs    = [5, 10, 20]
const B       = 100
const DIM     = 3
const M_grid  = collect(20:10:250)

generators = [
    ("Sinusoid", make_sinusoid),
    ("Sawtooth", make_sawtooth),
    ("Square",   make_square),
]

# ── Main simulation ────────────────────────────────────────────────────────────

println("=" ^ 70)
println("Period estimation: Topological (TopoTS.jl) vs FFT")
@printf("fs=%d Hz, T=%d samples, B=%d replicates\n", fs, T, B)
println("M_grid: $(first(M_grid))–$(last(M_grid)) step 10")
println("=" ^ 70)

results = Dict{String, Dict}()

for (wname, wgen) in generators
    results[wname] = Dict()
    for P in periods
        results[wname][P] = Dict()
        for snr in snrs
            errors_topo = Float64[]
            errors_fft  = Float64[]
            for b in 1:B
                noisy  = add_noise(wgen(P, T), snr)
                P_topo = topo_period_estimate(noisy, M_grid; dim=DIM)
                P_fft  = fft_period_estimate(noisy, fs)
                push!(errors_topo, P_topo - P)
                push!(errors_fft,  P_fft  - P)
            end
            rmse_t = sqrt(mean(errors_topo .^ 2))
            rmse_f = sqrt(mean(errors_fft  .^ 2))
            bias_t = mean(errors_topo)
            bias_f = mean(errors_fft)
            results[wname][P][snr] = (rmse_t, rmse_f, bias_t, bias_f)
            @printf("%-10s  P=%3d  SNR=%2d dB  |  Topo RMSE=%6.2f (bias=%+6.2f)  FFT RMSE=%6.2f (bias=%+6.2f)\n",
                    wname, P, snr, rmse_t, bias_t, rmse_f, bias_f)
        end
    end
end

# ── Summary table ──────────────────────────────────────────────────────────────

println()
println("=" ^ 70)
println("SUMMARY — winner declared when RMSE < 95% of opponent")
println("=" ^ 70)
@printf("%-10s  %6s  %5s  %8s  %8s  %-8s\n",
        "Waveform", "P", "SNR", "T-RMSE", "F-RMSE", "Winner")
println("-" ^ 70)
topo_wins = 0; fft_wins = 0; ties = 0
for (wname, _) in generators, P in periods, snr in snrs
    rt, rf, _, _ = results[wname][P][snr]
    if rt < rf * 0.95
        w = "TOPO"; topo_wins += 1
    elseif rf < rt * 0.95
        w = "FFT";  fft_wins  += 1
    else
        w = "tie";  ties      += 1
    end
    @printf("%-10s  %6d  %5d  %8.2f  %8.2f  %-8s\n", wname, P, snr, rt, rf, w)
end
println("=" ^ 70)
total = topo_wins + fft_wins + ties
@printf("\nTopo wins: %d/%d  |  FFT wins: %d/%d  |  Ties: %d/%d\n",
        topo_wins, total, fft_wins, total, ties, total)
verdict = if topo_wins >= fft_wins
    "KEEP §6.2 — topo competitive"
elseif fft_wins > topo_wins * 2
    "DELETE §6.2 — FFT dominates"
else
    "MARGINAL — topo wins on specific waveforms only; inspect breakdown"
end
println("Verdict: ", verdict)
