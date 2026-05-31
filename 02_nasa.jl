# =============================================================================
# 02_nasa.jl — NASA IMS bearing dataset regime detection
#
# Implements El-Yaagoubi et al. (2025) exactly:
#   Scenario 1: cumulative avg periodogram vs next trial (stationary case)
#     D_r = W₂(PH(mean(P₁..Pᵣ)), PH(P_{r+1}))
#   Scenario 2: consecutive periodograms (non-stationary/present vs next)
#     D_r = W₂(PH(Pᵣ), PH(P_{r+1}))
#
# Plus the novel phase-space pipeline:
#   Pipeline B: W₂(PH₁(embed(trialᵣ)), PH₁(embed(trialᵣ₊₁)))
#
# DATA (IMS README):
#   Set 1 — 2156 files, 8 channels
#     Bearing 1: ch 1 (x) — control, no failure
#     Bearing 3: ch 5 (x) — inner race defect      ← FAILS
#     Bearing 4: ch 7 (x) — roller element defect  ← FAILS
#   Set 2 — 984 files, 4 channels
#     Bearing 1: ch 1     — outer race failure      ← FAILS
#
# Usage: julia --threads=auto --project=. 02_nasa.jl C:\path\1st_test C:\path\2nd_test
# =============================================================================

using TopoTS, FFTW
using DelimitedFiles, Statistics, StatsBase
using CairoMakie, Printf
using Serialization

# =============================================================================
# Checkpoint helpers — save/load intermediate results to disk
# Checkpoint files live in checkpoints/ directory, named by bearing+step.
# =============================================================================

const CKPT_DIR = "checkpoints"
mkpath(CKPT_DIR)

ckpt_path(key::String) = joinpath(CKPT_DIR, key * ".jls")

"""
Load result from checkpoint if it exists, otherwise compute, save, and return.
`key` should be unique per bearing per step, e.g. "set1_b1_Ps".
"""
function cached(compute_fn::Function, key::String)
    path = ckpt_path(key)
    if isfile(path)
        println("      [checkpoint] loading $(basename(path)) ...")
        raw = deserialize(path)
        # Only attempt conversion for *_dgms files containing legacy diagram objects.
        # Periodogram files (_Ps) are Vector{Vector{Float64}} — leave them alone.
        # Distance files (_D) are Vector{Float64} — leave them alone.
        is_dgms_file = endswith(key, "_dgms")
        needs_conversion = is_dgms_file &&
            raw isa Vector && !isempty(raw) &&
            !(first(raw) isa Vector{Tuple{Float64,Float64}})
        if needs_conversion
            println("      [checkpoint] converting legacy diagram format ...")
            converted = map(raw) do d
                # DiagramCollection from persistent_homology: d[2] = H1 diagram
                # SublevelDiagram H0: d.H0 is Vector{Tuple}
                pairs_raw = try
                    endswith(key, "B_dgms") ? d[2] : d   # B = Rips H1, A2 = sublevel H0
                catch
                    d
                end
                [(Float64(b), Float64(dd))
                 for (b, dd) in pairs_raw if isfinite(Float64(dd))]
            end
            serialize(path, converted)
            println("      [checkpoint] resaved compact ($(filesize(path) ÷ 1024) KB)")
            return converted
        end
        return raw
    end
    result = compute_fn()
    serialize(path, result)
    println("      [checkpoint] saved $(basename(path)) ($(filesize(path) ÷ 1024) KB)")
    return result
end

mkpath("plots/empirical")

# =============================================================================
# 1. Data loading
# =============================================================================

function load_channel(dir::String, col::Int)
    # Filter: keep only regular files (no subdirs, no hidden files)
    # and only files that parse as data (non-zero size, no extension or numeric name)
    all_entries = readdir(dir; join=true)
    files = sort(filter(all_entries) do f
        !isdir(f) && filesize(f) > 0 && !startswith(basename(f), ".")
    end)
    isempty(files) && error("No files in $dir")
    @info "$(length(files)) trials, channel $col — $(basename(dir))"
    [readdlm(f, Float64)[:, col] for f in files]
end

# =============================================================================
# 2. Spectral features
# =============================================================================

# Subsample frequency axis: bearing fault frequencies < 5kHz, Nyquist ok at keep=4
# Reduces periodogram from 10241 → 2561 points: ~16x faster sublevel PH
function smoothed_periodogram(sig; bw=5, keep=4)
    P = abs2.(rfft(sig)) ./ length(sig)
    P = [mean(P[max(1,i-bw):min(length(P),i+bw)]) for i in 1:length(P)]
    return P[1:keep:end]
end

"""
Scenario 1 (El-Yaagoubi): cumulative average periodogram vs next trial.
D_r = W₂(PH(mean(P₁..Pᵣ)), PH(P_{r+1}))
Individual diagrams precomputed in parallel; cumulative avg loop sequential.
"""
function pipe_A_scenario1(ch::Vector{Vector{Float64}}, ck::String; bw=5)
    R  = length(ch)
    Ps = cached("$(ck)_A1_Ps") do
        result = Vector{Vector{Float64}}(undef, R)
        done   = Threads.Atomic{Int}(0)
        println("      precomputing periodograms (R=$R):")
        Threads.@threads for i in 1:R
            result[i] = smoothed_periodogram(ch[i]; bw)
            n = Threads.atomic_add!(done, 1) + 1
            print("\r        trial $n/$R"); flush(stdout)
        end
        println(); result
    end
    D = cached("$(ck)_A1_D") do
        result = zeros(R-1)
        S_sum  = copy(Ps[1])
        println("      cumulative L1 loop ($(R-1) steps):")
        for r in 1:R-1
            P_avg     = S_sum ./ r
            result[r] = sum(abs, P_avg .- Ps[r+1])
            S_sum    .+= Ps[r+1]
            print("\r        step $r/$(R-1)"); flush(stdout)
        end
        println(); result
    end
    return D
end

"""
Scenario 2 (El-Yaagoubi): consecutive periodograms.
D_r = W₂(PH(Pᵣ), PH(P_{r+1}))
All diagrams precomputed in parallel.
"""
function pipe_A_scenario2(ch::Vector{Vector{Float64}}, ck::String; bw=5)
    R    = length(ch)
    # Store only plain Vector{Tuple{Float64,Float64}} — not full diagram objects
    dgms = cached("$(ck)_A2_dgms") do
        result = Vector{Vector{Tuple{Float64,Float64}}}(undef, R)
        done   = Threads.Atomic{Int}(0)
        println("      precomputing diagrams (R=$R):")
        Threads.@threads for i in 1:R
            result[i] = [(Float64(b), Float64(d))
                         for (b,d) in sublevel_ph(smoothed_periodogram(ch[i]; bw)).H0]
            n = Threads.atomic_add!(done, 1) + 1
            print("\r        trial $n/$R"); flush(stdout)
        end
        println(); result
    end
    D = cached("$(ck)_A2_D") do
        result = zeros(R-1)
        done   = Threads.Atomic{Int}(0)
        println("      computing distances ($(R-1) pairs):")
        Threads.@threads for r in 1:R-1
            result[r] = wasserstein_distance(dgms[r], dgms[r+1])
            n = Threads.atomic_add!(done, 1) + 1
            print("\r        pair $n/$(R-1)"); flush(stdout)
        end
        println(); result
    end
    return D
end

# =============================================================================
# 3. Phase-space pipeline (novel)
#    Downsample each trial (20480→1024 pts at 1kHz), embed, Rips H₁
# =============================================================================

to_pairs(dgm) = [(Float64(b), Float64(d)) for (b,d) in dgm if isfinite(d)]

function pipe_B(ch::Vector{Vector{Float64}}, ck::String;
                sub::Int=20, n_ref::Int=10, dim::Int=3)
    ref = reduce(vcat, [c[1:sub:end] for c in ch[1:n_ref]])
    tau = optimal_lag(ref)
    @info "  Pipeline B: τ* = $tau ($(Threads.nthreads()) threads)"
    R    = length(ch)
    # Store only H1 birth-death pairs — not full DiagramCollection objects
    dgms = cached("$(ck)_B_dgms") do
        result = Vector{Vector{Tuple{Float64,Float64}}}(undef, R)
        done   = Threads.Atomic{Int}(0)
        println("      computing embeddings (R=$R):")
        Threads.@threads for i in eachindex(ch)
            ds  = ch[i][1:sub:end]
            emb = embed(ds; dim=dim, lag=tau)
            ph  = persistent_homology(emb; dim_max=1)
            result[i] = to_pairs(ph[2])   # extract H1 pairs immediately
            n = Threads.atomic_add!(done, 1) + 1
            print("\r        trial $n/$R"); flush(stdout)
        end
        println(); result
    end
    D = cached("$(ck)_B_D") do
        result = zeros(R-1)
        done   = Threads.Atomic{Int}(0)
        println("      computing H₁ distances ($(R-1) pairs):")
        Threads.@threads for r in 1:R-1
            result[r] = wasserstein_distance(dgms[r], dgms[r+1])
            n = Threads.atomic_add!(done, 1) + 1
            print("\r        pair $n/$(R-1)"); flush(stdout)
        end
        println(); result
    end
    return D
end

# =============================================================================
# 4. CUSUM with 3σ threshold (no null model available for physical system)
# =============================================================================

cusum(D, mu) = accumulate((c, d) -> max(0.0, c + d - mu), D; init=0.0)

function detect(D::Vector{Float64}; r0::Int)
    mu  = mean(D[1:r0]); sg = std(D[1:r0])
    τ   = mu + 3sg
    C   = cusum(D, mu)
    det = findfirst(>(τ), C)
    (tau=τ, mu=mu, C=C,
     detection=isnothing(det) ? nothing : det + 1)
end

# =============================================================================
# 5. Analyse one bearing channel — all three pipelines
# =============================================================================

function analyse(ch; r0, label, ck)
    R = length(ch)
    println("\n  $label  (R=$R, r₀=$r0 ≈ $(r0*10÷60) h)")

    println("    A Scenario 1 (cumul avg, L1):")
    D_A1  = pipe_A_scenario1(ch, ck)
    dA1   = detect(D_A1; r0)
    println("    → detected at trial $(dA1.detection)")

    println("    A Scenario 2 (consec, topo):")
    D_A2  = pipe_A_scenario2(ch, ck)
    dA2   = detect(D_A2; r0)
    println("    → detected at trial $(dA2.detection)")

    println("    B phase-space H₁:")
    D_B   = pipe_B(ch, ck)
    dB    = detect(D_B; r0)
    println("    → detected at trial $(dB.detection)")

    (D_A1=D_A1, D_A2=D_A2, D_B=D_B,
     det_A1=dA1, det_A2=dA2, det_B=dB)
end

# =============================================================================
# 6. Plot — normalised distance sequences for one bearing
# =============================================================================

function plot_bearing!(ax, r, r0; show_legend=false)
    # normalise each sequence to its own baseline mean
    norm(D, mu) = D ./ max(mu, 1e-12)

    D_A1n = norm(r.D_A1, r.det_A1.mu)
    D_A2n = norm(r.D_A2, r.det_A2.mu)
    D_Bn  = norm(r.D_B,  r.det_B.mu)
    τ_A1  = r.det_A1.tau / r.det_A1.mu
    τ_A2  = r.det_A2.tau / r.det_A2.mu
    τ_B   = r.det_B.tau  / r.det_B.mu

    idx = 2:length(D_A1n)+1
    lines!(ax, idx, D_A1n; color=:royalblue,   label="A Sc.1 (cumul avg)")
    lines!(ax, idx, D_A2n; color=:steelblue,   label="A Sc.2 (consec)",
           linestyle=:dash)
    lines!(ax, 2:length(D_Bn)+1, D_Bn;
           color=:darkorange, linewidth=2, label="B (phase-space H₁)")

    # Threshold lines omitted for the same reason

    # Detection markers omitted — visual interpretation only (cf. El-Yaagoubi et al. 2025)

    show_legend && axislegend(ax; position=:rt, framevisible=false,
                               labelsize=9)
end

# =============================================================================
# 7. Main
# =============================================================================

set1_dir = length(ARGS) >= 1 ? ARGS[1] : "data/1st_test"
set2_dir = length(ARGS) >= 2 ? ARGS[2] : "data/2nd_test"
for d in (set1_dir, set2_dir)
    isdir(d) || error("Not found: $d\nUsage: julia 02_nasa.jl /path/1st_test /path/2nd_test")
end

# --- Set 1 ---
println("\n===== Set 1 (2156 trials) =====")
R1   = length(filter(!isdir, readdir(set1_dir)))
r0_1 = div(R1, 10)

res1_b1 = analyse(load_channel(set1_dir, 1); r0=r0_1,
                  label="Bearing 1 — control", ck="set1_b1")
res1_b3 = analyse(load_channel(set1_dir, 5); r0=r0_1,
                  label="Bearing 3 — inner race defect", ck="set1_b3")
res1_b4 = analyse(load_channel(set1_dir, 7); r0=r0_1,
                  label="Bearing 4 — roller element defect", ck="set1_b4")

fig1 = Figure(size=(950, 950))
for (row, (r, title)) in enumerate([
        (res1_b1, "Set 1 · Bearing 1 — control"),
        (res1_b3, "Set 1 · Bearing 3 — inner race defect"),
        (res1_b4, "Set 1 · Bearing 4 — roller element defect")])
    ax = Axis(fig1[row, 1]; title=title,
              xlabel="Trial r", ylabel="W₂ / baseline mean")
    plot_bearing!(ax, r, r0_1; show_legend=(row==1))
end
save("plots/empirical/nasa_set1.pdf", fig1)
println("\nSaved → plots/empirical/nasa_set1.pdf")

# --- Set 2 ---
println("\n===== Set 2 (984 trials) =====")
R2   = length(filter(!isdir, readdir(set2_dir)))
r0_2 = div(R2, 10)

res2_b1 = analyse(load_channel(set2_dir, 1); r0=r0_2,
                  label="Bearing 1 — outer race failure", ck="set2_b1")

fig2 = Figure(size=(950, 400))
ax = Axis(fig2[1, 1]; title="Set 2 · Bearing 1 — outer race failure",
          xlabel="Trial r", ylabel="W₂ / baseline mean")
plot_bearing!(ax, res2_b1, r0_2; show_legend=true)
save("plots/empirical/nasa_set2.pdf", fig2)
println("Saved → plots/empirical/nasa_set2.pdf")

# --- Summary ---
println("\n=== Detection summary ===")
@printf("  %-26s  %-14s %-14s %-14s\n", "Bearing", "A-Sc1 (L1)", "A-Sc2 (topo)", "B (phase-space)")
println("  " * "-"^72)
fmt(x) = isnothing(x) ? "none          " : @sprintf("trial %-8d", x)
for (lbl, r) in [("Set1 B1 control",    res1_b1),
                  ("Set1 B3 inner race", res1_b3),
                  ("Set1 B4 roller",     res1_b4),
                  ("Set2 B1 outer race", res2_b1)]
    @printf("  %-26s  %s %s %s\n", lbl,
            fmt(r.det_A1.detection),
            fmt(r.det_A2.detection),
            fmt(r.det_B.detection))
end

# --- Set 3 ---
# Set 3: 4448 files, 4 channels (one per bearing)
#   Bearing 1: ch 1 — no failure (control)
#   Bearing 2: ch 2 — no failure
#   Bearing 3: ch 3 — outer race failure  ← FAILS
#   Bearing 4: ch 4 — no failure
set3_dir = length(ARGS) >= 3 ? ARGS[3] : "data/3rd_test"
if isdir(set3_dir)
    println("\n===== Set 3 (4448 trials) =====")
    R3   = length(filter(!isdir, readdir(set3_dir)))
    r0_3 = div(R3, 10)

    res3_b1 = analyse(load_channel(set3_dir, 1); r0=r0_3,
                      label="Bearing 1 — control", ck="set3_b1")
    res3_b3 = analyse(load_channel(set3_dir, 3); r0=r0_3,
                      label="Bearing 3 — outer race failure", ck="set3_b3")

    fig3 = Figure(size=(950, 650))
    for (row, (r, title)) in enumerate([
            (res3_b1, "Set 3 · Bearing 1 — control"),
            (res3_b3, "Set 3 · Bearing 3 — outer race failure")])
        ax = Axis(fig3[row, 1]; title=title,
                  xlabel="Trial r", ylabel="W₂ / baseline mean")
        plot_bearing!(ax, r, r0_3; show_legend=(row==1))
    end
    save("plots/empirical/nasa_set3.pdf", fig3)
    println("Saved → plots/empirical/nasa_set3.pdf")

    # Append Set 3 to summary
    println("\n=== Set 3 detection ===")
    fmt3(x) = isnothing(x) ? "none          " : @sprintf("trial %-8d", x)
    for (lbl, r) in [("Set3 B1 control",    res3_b1),
                      ("Set3 B3 outer race", res3_b3)]
        @printf("  %-26s  %s %s %s\n", lbl,
                fmt3(r.det_A1.detection),
                fmt3(r.det_A2.detection),
                fmt3(r.det_B.detection))
    end
else
    @warn "Set 3 directory not found: $set3_dir — skipping"
end
