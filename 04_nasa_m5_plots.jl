# =============================================================================
# 04_nasa_m5_plots.jl — NASA figures with M5 offline change-point marker
#
# Loads precomputed distance sequences from checkpoints.
# Applies M5 (offline RSS minimisation) to find the change-point.
# Plots clean distance sequences with a single vertical line at r*.
#
# Run: julia --project=. 04_nasa_m5_plots.jl
# =============================================================================

using Serialization, Statistics
using CairoMakie, Printf
using TopoTS

mkpath("plots/empirical")

const CKPT_DIR = "checkpoints"

# =============================================================================
# 1. Load and detect
# =============================================================================

function load_D(key)
    p = joinpath(CKPT_DIR, key * ".jls")
    isfile(p) || error("Not found: $p — run 02_nasa.jl first")
    deserialize(p)
end

# =============================================================================
# 2. Plot one bearing panel
# =============================================================================

function bearing_panel!(ax, D_A1, D_A2, D_B,
                        r_A1, r_A2, r_B,
                        sig_A1, sig_A2, sig_B, r0;
                        show_legend=false)
    # Normalise to baseline mean
    mu_A1 = mean(D_A1[1:r0]); mu_A2 = mean(D_A2[1:r0])
    mu_B  = mean(D_B[1:r0])
    N = length(D_A1)
    idx = 2:N+1

    lines!(ax, idx, D_A1 ./ mu_A1; color=:royalblue,
           label="A Sc.1 (cumul L1)")
    lines!(ax, idx, D_A2 ./ mu_A2; color=:steelblue,
           linestyle=:dash, label="A Sc.2 (topo H₀)")
    lines!(ax, 2:length(D_B)+1, D_B ./ mu_B;
           color=:darkorange, linewidth=2, label="B (phase-space H₁)")

    # Draw three separate M5 change-point markers, slightly offset vertically
    # so they remain distinguishable even when r* values nearly coincide.
    # Each marker is a short thick tick at the top of the axis for its pipeline.
    R_total = length(D_A1)
    late(r) = r / R_total > 0.6
    failure(sig, r) = sig && late(r)


    # Small x-offsets so lines remain distinguishable when r* values coincide
    # Offsets in trial units: A-Sc1 at r-2, A-Sc2 at r, B at r+2
    for (r, sig, col, offset) in [
            (r_A1, sig_A1, :royalblue,  -3),
            (r_A2, sig_A2, :steelblue,   0),
            (r_B,  sig_B,  :darkorange, +3)]
        d  = failure(sig, r)
        # White backing for visibility
        vlines!(ax, [r + offset]; color=(:white, 0.6), linewidth=5.0)
        vlines!(ax, [r + offset];
                color=(col, 1.0),
                linestyle = d ? :solid : :dot,
                linewidth  = d ? 2.5 : 1.5)
    end

    if show_legend
        axislegend(ax; position=:rt, framevisible=false, labelsize=9)
        # Annotate line style meaning
        text!(ax, 0.02, 0.97;
              text="Vertical lines: M5 change-point
(solid = significant, dotted = not)",
              align=(:left, :top), space=:relative, fontsize=8,
              color=:gray40)
    end
end

# =============================================================================
# 3. Analyse one bearing
# =============================================================================

function analyse(ck; r0, label)
    D_A1 = load_D("$(ck)_A1_D")
    D_A2 = load_D("$(ck)_A2_D")
    D_B  = load_D("$(ck)_B_D")
    res_A1 = andrews_supF(D_A1; r0=r0, alpha=0.05)
    r_A1 = res_A1.r_star
    sig_A1 = res_A1.significant
    res_A2 = andrews_supF(D_A2; r0=r0, alpha=0.05)
    r_A2 = res_A2.r_star
    sig_A2 = res_A2.significant
    res_B  = andrews_supF(D_B;  r0=r0, alpha=0.05)
    r_B  = res_B.r_star
    sig_B  = res_B.significant
    R = length(D_A1) + 1
    star(s) = s ? "*" : " "
    @printf("  %-36s  A-Sc1: %4d/%d (%.2f)%s  A-Sc2: %4d/%d (%.2f)%s  B: %4d/%d (%.2f)%s\n",
            label,
            r_A1, R, r_A1/R, star(sig_A1),
            r_A2, R, r_A2/R, star(sig_A2),
            r_B,  R, r_B/R,  star(sig_B))
    (D_A1=D_A1, D_A2=D_A2, D_B=D_B,
     r_A1=r_A1, r_A2=r_A2, r_B=r_B,
     sig_A1=sig_A1, sig_A2=sig_A2, sig_B=sig_B,
     r0=r0, label=label, R=R)
end

# =============================================================================
# 4. Main
# =============================================================================

println("=== M5 offline change-point detection ===\n")
println("Format: trial / total (fraction)")

# --- Set 1 ---
println("\nSet 1 (R=2156, r₀=215):")
r0_1 = 215
b1 = analyse("set1_b1"; r0=r0_1, label="Bearing 1 — control")
b3 = analyse("set1_b3"; r0=r0_1, label="Bearing 3 — inner race defect")
b4 = analyse("set1_b4"; r0=r0_1, label="Bearing 4 — roller element defect")

fig1 = Figure(size=(980, 980))
for (row, (br, title)) in enumerate([
        (b1, "Set 1 · Bearing 1 — control"),
        (b3, "Set 1 · Bearing 3 — inner race defect"),
        (b4, "Set 1 · Bearing 4 — roller element defect")])
    ax = Axis(fig1[row, 1]; title=title,
              xlabel="Trial r", ylabel="W₂ / baseline mean")
    bearing_panel!(ax, br.D_A1, br.D_A2, br.D_B,
                   br.r_A1, br.r_A2, br.r_B,
                   br.sig_A1, br.sig_A2, br.sig_B, br.r0;
                   show_legend=(row==1))
end
save("plots/empirical/nasa_set1_m5.pdf", fig1)
println("\nSaved → plots/empirical/nasa_set1_m5.pdf")

# --- Set 2 ---
println("\nSet 2 (R=984, r₀=98):")
r0_2 = 98
b2_1 = analyse("set2_b1"; r0=r0_2, label="Bearing 1 — outer race failure")

fig2 = Figure(size=(980, 480))
ax = Axis(fig2[1, 1];
          title="Set 2 · Bearing 1 — outer race failure",
          xlabel="Trial r", ylabel="W₂ / baseline mean")
bearing_panel!(ax, b2_1.D_A1, b2_1.D_A2, b2_1.D_B,
               b2_1.r_A1, b2_1.r_A2, b2_1.r_B,
               b2_1.sig_A1, b2_1.sig_A2, b2_1.sig_B, r0_2;
               show_legend=true)
save("plots/empirical/nasa_set2_m5.pdf", fig2)
println("Saved → plots/empirical/nasa_set2_m5.pdf")

# --- Set 3 ---
println("\nSet 3 (R=6324, r₀=632):")
r0_3 = 632
b3_1 = analyse("set3_b1"; r0=r0_3, label="Bearing 1 — control")
b3_3 = analyse("set3_b3"; r0=r0_3, label="Bearing 3 — outer race failure")

fig3 = Figure(size=(980, 680))
for (row, (br, title)) in enumerate([
        (b3_1, "Set 3 · Bearing 1 — control"),
        (b3_3, "Set 3 · Bearing 3 — outer race failure")])
    ax = Axis(fig3[row, 1]; title=title,
              xlabel="Trial r", ylabel="W₂ / baseline mean")
    bearing_panel!(ax, br.D_A1, br.D_A2, br.D_B,
                   br.r_A1, br.r_A2, br.r_B,
                   br.sig_A1, br.sig_A2, br.sig_B, br.r0;
                   show_legend=(row==1))
end
save("plots/empirical/nasa_set3_m5.pdf", fig3)
println("Saved → plots/empirical/nasa_set3_m5.pdf")

# =============================================================================
# 5. Summary
# =============================================================================

println("\n=== Summary: r*/R by bearing ===")
println("  Control bearings should have r*/R << 1 (random mid-experiment)")
println("  Failing  bearings should have r*/R close to 1 (late-stage shift)")
println()
@printf("  %-38s  %6s %6s %6s\n", "Bearing", "A-Sc1", "A-Sc2", "B")
println("  " * "-"^60)
println("  Decision rule: significant (Andrews sup-F) AND r*/R > 0.6")
println("  * = failure declared  · = significant but early  — = not significant")
@printf("  %-38s  %8s %8s %8s\n", "Bearing", "A-Sc1", "A-Sc2", "B")
println("  " * "-"^66)
for br in [b1, b3, b4, b2_1, b3_1, b3_3]
    r_con = round(Int, median([br.r_A1, br.r_A2, br.r_B]))
    sig_con = br.sig_A1 || br.sig_A2 || br.sig_B
    late_con = r_con / br.R > 0.6
    verdict = !sig_con ? "—" : (!late_con ? "· early" : "* FAILURE")
    @printf("  %-38s  r*=%4d/%-5d (%.2f)  %s\n",
            br.label, r_con, br.R, r_con/br.R, verdict)
end
