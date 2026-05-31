# =============================================================================
# 05_nasa_crocker.jl — CROCKER plots from existing Pipeline B checkpoints
#
# Loads H₁ persistence diagram pairs already computed by 02_nasa.jl
# (stored in checkpoints/*_B_dgms.jls) and computes the CROCKER matrix
# β₁(ε, r) without any recomputation of embeddings or PH.
#
# CROCKER matrix: C[j, r] = #{(b,d) ∈ Dgm_r : b ≤ ε_j ≤ d}
#
# Run: julia --project=. 05_nasa_crocker.jl
# (Fast — no heavy computation, just loading checkpoints)
# =============================================================================

using Serialization, Statistics
using CairoMakie, Printf

mkpath("plots/empirical")
mkpath("checkpoints")

const CKPT_DIR = "checkpoints"

ckpt_path(key) = joinpath(CKPT_DIR, key * ".jls")

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
# 1. Load existing H₁ diagram checkpoints
# =============================================================================

function load_dgms(ck::String)
    p = ckpt_path("$(ck)_B_dgms")
    isfile(p) || error("Checkpoint not found: $p — run 02_nasa.jl first")
    raw = deserialize(p)
    # Handle both legacy (full DiagramCollection) and compact (tuple pairs) formats
    if raw isa Vector && !isempty(raw) && first(raw) isa Vector{Tuple{Float64,Float64}}
        return raw   # already compact
    end
    # Legacy: extract H₁ pairs
    println("  converting legacy format...")
    return [[(Float64(b), Float64(d)) for (b,d) in (try d[2] catch; d end)
              if isfinite(Float64(d))]
            for d in raw]
end

# =============================================================================
# 2. Compute CROCKER matrix from H₁ diagrams
#    C[j, r] = β₁(ε_j, r) = #{(b,d) : b ≤ ε_j ≤ d}
# =============================================================================

function crocker_matrix(dgms::Vector{Vector{Tuple{Float64,Float64}}};
                        n_eps=60, trim_lo=0.05, trim_hi=0.95)
    # Global ε grid from all birth/death values
    all_b = [b for dgm in dgms for (b,_) in dgm]
    all_d = [d for dgm in dgms for (_,d) in dgm]

    if isempty(all_b)
        @warn "No H₁ features — returning empty matrix"
        return zeros(Int, n_eps, length(dgms)), range(0, 1; length=n_eps)
    end

    eps_min = quantile(all_b, trim_lo)
    eps_max = quantile(all_d, trim_hi)
    eps_grid = range(eps_min, eps_max; length=n_eps)

    R = length(dgms)
    C = zeros(Int, n_eps, R)
    for r in 1:R
        dgm = dgms[r]
        isempty(dgm) && continue
        for (j, ε) in enumerate(eps_grid)
            C[j, r] = count(((b,d),) -> b ≤ ε ≤ d, dgm)
        end
    end
    return C, eps_grid
end

# =============================================================================
# 3. Plot CROCKER heatmap panel
# =============================================================================

function crocker_panel!(ax, C, eps_grid; title="", r_star=nothing,
                        clims=nothing)
    R      = size(C, 2)
    crange = isnothing(clims) ? (0, max(1, maximum(C))) : clims
    hm = heatmap!(ax, 1:R, collect(eps_grid), C';
                  colormap=Reverse(:deep),
                  colorrange=crange,
                  interpolate=false)
    ax.title  = title
    ax.xlabel = "Trial r"
    ax.ylabel = "Filtration radius ε"
    if !isnothing(r_star)
        vlines!(ax, [r_star]; color=:white, linewidth=2.0,
                linestyle=:dash, label="M5 r*")
    end
    return hm
end

# =============================================================================
# 4. Main
# =============================================================================

println("=== CROCKER plots from Pipeline B checkpoints ===\n")

# M5 consensus change-points (from 04_nasa_m5_plots.jl)
R_STAR = Dict(
    "set1_b1" => nothing,   # control — no failure
    "set1_b3" => 1939,      # inner race
    "set1_b4" => 1614,      # roller element
    "set2_b1" => 885,       # outer race
    "set3_b1" => nothing,   # control — ambiguous
    "set3_b3" => 5691,      # outer race
)

TITLES = Dict(
    "set1_b1" => "Set 1 · Bearing 1 — control",
    "set1_b3" => "Set 1 · Bearing 3 — inner race defect",
    "set1_b4" => "Set 1 · Bearing 4 — roller element defect",
    "set2_b1" => "Set 2 · Bearing 1 — outer race failure",
    "set3_b1" => "Set 3 · Bearing 1 — control",
    "set3_b3" => "Set 3 · Bearing 3 — outer race failure",
)

# Compute and cache CROCKER matrices
CROCKER = Dict{String, Tuple}()
for ck in keys(R_STAR)
    print("Loading $ck ... "); flush(stdout)
    dgms = load_dgms(ck)
    C, eps = cached("$(ck)_crocker_matrix") do
        println("computing CROCKER matrix ($(length(dgms)) trials)...")
        crocker_matrix(dgms)
    end
    CROCKER[ck] = (C, eps)
    println("  β₁ range: 0–$(maximum(C)), R=$(size(C,2))")
end

# =============================================================================
# 5. Figure 1: Set 1 — control vs inner race vs roller element
# =============================================================================

fig1 = Figure(size=(1400, 1100))

# Shared color limits across Set 1 panels
clim1 = (0, maximum(maximum(CROCKER[k][1]) for k in ("set1_b1","set1_b3","set1_b4")))

for (row, ck) in enumerate(("set1_b1", "set1_b3", "set1_b4"))
    C, eps = CROCKER[ck]
    ax = Axis(fig1[row, 1])
    hm = crocker_panel!(ax, C, eps;
                        title=TITLES[ck],
                        r_star=R_STAR[ck],
                        clims=clim1)
    row == 1 && Colorbar(fig1[row, 2], hm; label="β₁(ε, r)")
end

save("plots/empirical/crocker_set1.pdf", fig1)
println("\nSaved → plots/empirical/crocker_set1.pdf")

# =============================================================================
# 6. Figure 2: Set 2 — outer race failure
# =============================================================================

fig2 = Figure(size=(900, 500))
C2, eps2 = CROCKER["set2_b1"]
ax2 = Axis(fig2[1, 1])
hm2 = crocker_panel!(ax2, C2, eps2;
                     title=TITLES["set2_b1"],
                     r_star=R_STAR["set2_b1"])
Colorbar(fig2[1, 2], hm2; label="β₁(ε, r)")
save("plots/empirical/crocker_set2.pdf", fig2)
println("Saved → plots/empirical/crocker_set2.pdf")

# =============================================================================
# 7. Figure 3: Set 3 — control vs outer race failure
# =============================================================================

fig3 = Figure(size=(1400, 750))
clim3 = (0, maximum(maximum(CROCKER[k][1]) for k in ("set3_b1","set3_b3")))

for (row, ck) in enumerate(("set3_b1", "set3_b3"))
    C, eps = CROCKER[ck]
    ax = Axis(fig3[row, 1])
    hm = crocker_panel!(ax, C, eps;
                        title=TITLES[ck],
                        r_star=R_STAR[ck],
                        clims=clim3)
    row == 1 && Colorbar(fig3[row, 2], hm; label="β₁(ε, r)")
end
save("plots/empirical/crocker_set3.pdf", fig3)
println("Saved → plots/empirical/crocker_set3.pdf")

println("\nAll CROCKER figures saved to plots/empirical/")
