# =============================================================================
# 03_nasa_detection.jl — NASA detection threshold comparison
#
# Loads precomputed distance sequences from checkpoints (no recomputation).
# Tests four threshold/detection methods on all bearings:
#
#   M1: Standard CUSUM, 3σ threshold (baseline — what we had before)
#   M2: Sustained CUSUM, 3σ threshold, k=10 consecutive exceedances
#   M3: Standard CUSUM, 99th percentile threshold from burn-in
#   M4: Adaptive CUSUM — baseline estimated in sliding window, 3σ
#
# Prints a full comparison table for all bearings × all methods.
# Run: julia --project=. 03_nasa_detection.jl
# =============================================================================

using Serialization, Statistics, StatsBase
using CairoMakie, Printf

mkpath("plots/empirical")

const CKPT_DIR = "checkpoints"

# =============================================================================
# 1. Load precomputed distance sequences
# =============================================================================

function load_D(key::String)
    p = joinpath(CKPT_DIR, key * ".jls")
    isfile(p) || error("Checkpoint not found: $p\nRun 02_nasa.jl first.")
    return deserialize(p)
end

# =============================================================================
# 2. Four detection methods
# =============================================================================

"""M1: Standard CUSUM with 3σ threshold. Fires on first exceedance."""
function detect_M1(D::Vector{Float64}; r0::Int, n_sigma=3.0)
    mu = mean(D[1:r0]); sg = std(D[1:r0])
    τ  = mu + n_sigma * sg
    C  = cusum(D, mu)
    det = findfirst(>(τ), @view C[r0+1:end])
    (detection=isnothing(det) ? nothing : det + r0, tau=τ, mu=mu, C=C)
end

"""M2: Sustained CUSUM — requires k consecutive exceedances of 3σ threshold."""
function detect_M2(D::Vector{Float64}; r0::Int, k=10, n_sigma=3.0)
    mu = mean(D[1:r0]); sg = std(D[1:r0])
    τ  = mu + n_sigma * sg
    C  = cusum(D, mu)
    consec = 0
    for r in r0+1:length(C)
        C[r] > τ ? (consec += 1) : (consec = 0)
        consec >= k && return (detection=r-k+1, tau=τ, mu=mu, C=C)
    end
    (detection=nothing, tau=τ, mu=mu, C=C)
end

"""M3: Standard CUSUM with 99th percentile threshold from burn-in distances."""
function detect_M3(D::Vector{Float64}; r0::Int)
    mu = mean(D[1:r0])
    τ  = quantile(D[1:r0], 0.99)
    # Build CUSUM relative to baseline but threshold on raw D values
    C  = cusum(D, mu)
    # Threshold on CUSUM using scale derived from percentile
    τ_C = τ - mu   # how far above mean the 99th pct sits
    det = findfirst(>(τ_C * r0), @view C[r0+1:end])  # require accumulation
    # Simpler: just threshold on raw D exceeding 99th pct of burn-in
    det_raw = findfirst(>(τ), @view D[r0+1:end])
    (detection=isnothing(det_raw) ? nothing : det_raw + r0,
     tau=τ, mu=mu, C=C)
end

"""
M4: Adaptive CUSUM — sliding window of width `win` estimates local baseline.
Fires when local deviation exceeds n_sigma × local std.
"""
function detect_M4(D::Vector{Float64}; r0::Int, win=50, n_sigma=3.0)
    C   = zeros(length(D))
    det = nothing
    for r in r0+1:length(D)
        lo  = max(1, r - win)
        mu  = mean(D[lo:r-1])
        sg  = std(D[lo:r-1])
        sg  = max(sg, 1e-12)
        τ   = mu + n_sigma * sg
        C[r] = max(0.0, C[r-1] + D[r] - mu)
        if D[r] > τ && isnothing(det)
            det = r
        end
    end
    mu0 = mean(D[1:r0]); sg0 = std(D[1:r0])
    (detection=det, tau=mu0 + n_sigma*sg0, mu=mu0, C=C)
end

cusum(D, mu) = accumulate((c, d) -> max(0.0, c + d - mu), D; init=0.0)

"""
M5: Offline RSS minimisation — finds the single best level-shift change-point.
No threshold, no burn-in, no false-alarm rate.
Minimises within-segment sum of squared deviations from segment mean.
Returns the trial r* that best separates two constant-level segments.
"""
function detect_M5(D::Vector{Float64}; r0::Int, kwargs...)
    R   = length(D)
    # Search only after burn-in — pre-change segment must have >= r0 points
    best_rss = Inf
    best_r   = nothing
    # Precompute prefix sums for O(n) evaluation
    S  = cumsum(D)
    S2 = cumsum(D .^ 2)
    total_mean = S[R] / R
    for r in r0:R-r0
        n1   = r
        n2   = R - r
        mu1  = S[r] / n1
        mu2  = (S[R] - S[r]) / n2
        rss1 = S2[r] - n1 * mu1^2
        rss2 = (S2[R] - S2[r]) - n2 * mu2^2
        rss  = rss1 + rss2
        if rss < best_rss
            best_rss = rss
            best_r   = r
        end
    end
    # Use a dummy threshold/mu for interface compatibility
    mu = mean(D[1:r0])
    sg = std(D[1:r0])
    C  = cusum(D, mu)   # kept for plot compatibility
    (detection=best_r, tau=mu+3sg, mu=mu, C=C, rss=best_rss)
end

"""
M6: Andrews (1993) sup-F test for structural change.
Computes the F-statistic at every candidate change-point and compares the
supremum against asymptotic critical values from Andrews (1993), Table 1
(15% trimming, heteroskedasticity-robust).

H₀: no structural change (constant mean throughout)
H₁: single level shift at unknown location

Critical values (Andrews 1993, 1 restriction, 15% trimming):
  α = 0.10 → 7.12
  α = 0.05 → 8.85
  α = 0.01 → 12.16

Returns the location of the sup-F (same as M5) and the p-value
approximated from the asymptotic distribution.
Rejection of H₀ means a genuine structural change is detected.
"""
function detect_M6(D::Vector{Float64}; r0::Int, alpha=0.05, kwargs...)
    R    = length(D)
    S    = cumsum(D)
    S2   = cumsum(D .^ 2)

    # Null RSS (single segment)
    mu_null   = S[R] / R
    rss_null  = S2[R] - R * mu_null^2

    best_F  = -Inf
    best_r  = div(R, 2)
    F_vals  = zeros(R)

    for r in r0:R-r0
        n1  = r;      n2  = R - r
        mu1 = S[r] / n1
        mu2 = (S[R] - S[r]) / n2
        rss = (S2[r] - n1*mu1^2) + ((S2[R]-S2[r]) - n2*mu2^2)
        # F-statistic: improvement from splitting / residual variance
        F_vals[r] = ((rss_null - rss) / 2) / (rss / (R - 2))
        if F_vals[r] > best_F
            best_F = F_vals[r]
            best_r = r
        end
    end

    # Andrews (1993) asymptotic critical values, 1 restriction, 15% trim
    cv = Dict(0.10 => 7.12, 0.05 => 8.85, 0.01 => 12.16)
    cv_alpha = get(cv, alpha, 8.85)

    # Detection declared if sup-F exceeds critical value
    detected = best_F > cv_alpha ? best_r : nothing

    mu = mean(D[1:r0]); sg = std(D[1:r0])
    C  = cusum(D, mu)
    (detection=detected, tau=cv_alpha, mu=mu, C=C,
     sup_F=best_F, cv=cv_alpha, F_vals=F_vals)
end

METHODS = [
    ("M1", "Standard CUSUM (3σ)",        detect_M1),
    ("M2", "Sustained CUSUM (3σ, k=10)", detect_M2),
    ("M3", "99th pct threshold",          detect_M3),
    ("M4", "Adaptive CUSUM (3σ, w=50)",  detect_M4),
    ("M5", "Offline RSS minimisation",    detect_M5),
    ("M6", "Andrews sup-F (α=0.05)",      detect_M6),
]

# =============================================================================
# 3. Analyse one bearing across all four methods
# =============================================================================

function analyse(ck::String; r0::Int, label::String)
    D_A1 = load_D("$(ck)_A1_D")
    D_A2 = load_D("$(ck)_A2_D")
    D_B  = load_D("$(ck)_B_D")

    results = Dict{String, NamedTuple}()
    for (mkey, _, dfn) in METHODS
        results["$(mkey)_A1"] = dfn(D_A1; r0)
        results["$(mkey)_A2"] = dfn(D_A2; r0)
        results["$(mkey)_B"]  = dfn(D_B;  r0)
    end
    return (D_A1=D_A1, D_A2=D_A2, D_B=D_B,
            r0=r0, label=label, results=results)
end

fmt(x) = isnothing(x) ? "   —  " : @sprintf("%4d  ", x)

function print_bearing(br)
    println("\n  $(br.label)")
    @printf("    %-28s  %-8s %-8s %-8s\n", "Method", "A-Sc1", "A-Sc2", "B")
    println("    " * "-"^52)
    for (mkey, mlabel, _) in METHODS
        @printf("    %-28s  %s %s %s\n", mlabel,
                fmt(br.results["$(mkey)_A1"].detection),
                fmt(br.results["$(mkey)_A2"].detection),
                fmt(br.results["$(mkey)_B"].detection))
    end
end

# =============================================================================
# 4. Run all bearings
# =============================================================================

BEARINGS = [
    ("set1_b1", 215,  "Set1 B1 — control (no failure)"),
    ("set1_b3", 215,  "Set1 B3 — inner race defect"),
    ("set1_b4", 215,  "Set1 B4 — roller element defect"),
    ("set2_b1", 98,   "Set2 B1 — outer race failure"),
    ("set3_b1", 632,  "Set3 B1 — control (no failure)"),
    ("set3_b3", 632,  "Set3 B3 — outer race failure"),
]

println("=== NASA IMS: detection method comparison ===\n")
println("Columns: detection trial for each pipeline")
println("  A-Sc1 = cumulative L1 periodogram")
println("  A-Sc2 = consecutive topological H₀ (El-Yaagoubi)")
println("  B     = phase-space H₁ (Takens)")

ALL = []
for (ck, r0, label) in BEARINGS
    br = analyse(ck; r0, label)
    print_bearing(br)
    push!(ALL, br)
end

# =============================================================================
# 5. LaTeX table
# =============================================================================

println("\n\n=== LaTeX table ===")
println("""
\\begin{table}[ht]
\\centering
\\caption{Detection trials for each pipeline and threshold method on the
  NASA IMS bearing dataset. Each trial corresponds to 10 minutes of
  operation. Dashes indicate no detection within the experiment duration.}
\\label{tab:nasa-detection}
\\small
\\begin{tabular}{llccc}
\\toprule
Bearing & Method & A-Sc1 (L1) & A-Sc2 (topo) & B (phase) \\\\
\\midrule""")

for br in ALL
    first_row = true
    bname = replace(br.label, r" — .*" => "")
    for (mkey, mlabel, _) in METHODS
        dA1 = br.results["$(mkey)_A1"].detection
        dA2 = br.results["$(mkey)_A2"].detection
        dB  = br.results["$(mkey)_B"].detection
        fl(x) = isnothing(x) ? "—" : string(x)
        if first_row
            @printf("\\multirow{6}{*}{%s} & %s & %s & %s & %s \\\\\n",
                    bname, mlabel, fl(dA1), fl(dA2), fl(dB))
            first_row = false
        else
            @printf(" & %s & %s & %s & %s \\\\\n",
                    mlabel, fl(dA1), fl(dA2), fl(dB))
        end
    end
    println("\\midrule")
end

println("""\\bottomrule
\\end{tabular}
\\end{table}""")

# =============================================================================
# 7. Relative position summary — r*/R for M5 (key diagnostic)
# =============================================================================

println("\n=== M5 offline change-point: relative position r*/R ===")
println("  (closer to 1.0 = later in experiment = more likely genuine failure)")
@printf("  %-38s  %8s %8s %8s\n", "Bearing", "A-Sc1", "A-Sc2", "B")
println("  " * "-"^68)
for br in ALL
    R = length(br.D_A1) + 1
    d5A1 = br.results["M5_A1"].detection
    d5A2 = br.results["M5_A2"].detection
    d5B  = br.results["M5_B"].detection
    frac(x) = isnothing(x) ? "   —     " : @sprintf("%4d/%-5d", x, R)
    @printf("  %-38s  %s %s %s\n", br.label,
            frac(d5A1), frac(d5A2), frac(d5B))
end

println("\n=== M6 Andrews sup-F test (α=0.05, cv=8.85) ===")
println("  H₀: no structural change.  Detection = sup-F > 8.85")
println("  sup-F value shown; * = significant at 5%")
@printf("  %-38s  %12s %12s %12s\n",
        "Bearing", "A-Sc1 sup-F", "A-Sc2 sup-F", "B sup-F")
println("  " * "-"^80)
for br in ALL
    function fmt6(key)
        r = br.results[key]
        sig = r.sup_F > r.cv ? "*" : " "
        @sprintf("%8.2f%s", r.sup_F, sig)
    end
    @printf("  %-38s  %12s %12s %12s\n",
            br.label, fmt6("M6_A1"), fmt6("M6_A2"), fmt6("M6_B"))
end
