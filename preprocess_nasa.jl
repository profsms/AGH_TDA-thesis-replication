##############################################################################
# preprocess_nasa.jl
#
# Run this script ONCE on the raw NASA IMS bearing dataset to produce
# compact preprocessed files that can be bundled with the replication
# repository.
#
# INPUT:  Raw IMS data in IMS/1st_test/, IMS/2nd_test/, IMS/3rd_test/
#         (see README.md for download and setup instructions)
# OUTPUT: data/nasa_set1.jls  (~50 MB before compression)
#         data/nasa_set2.jls  (~8  MB)
#         data/nasa_set3.jls  (~100 MB)
#
# What is kept per experiment:
#   Set 1 (8 ch): channels 1 (B1 control), 5 (B3 inner race),
#                           7 (B4 roller element) — downsampled ×20
#   Set 2 (4 ch): channel 1 (B1 outer race) — downsampled ×20
#   Set 3 (4 ch): channels 1 (B1 control), 3 (B3 outer race) — downsampled ×20
#
# USAGE
#   julia preprocess_nasa.jl
#   julia preprocess_nasa.jl /path/1st_test /path/2nd_test /path/3rd_test
##############################################################################

using DelimitedFiles, Serialization, Printf

const SUB = 20   # downsample factor

set1_dir = get(ENV, "IMS_SET1", length(ARGS) >= 1 ? ARGS[1] : "IMS/1st_test")
set2_dir = get(ENV, "IMS_SET2", length(ARGS) >= 2 ? ARGS[2] : "IMS/2nd_test")
set3_dir = get(ENV, "IMS_SET3", length(ARGS) >= 3 ? ARGS[3] : "IMS/3rd_test")

mkpath("data")

function trial_files(dir::String)
    sort(filter(f -> !isdir(f) && filesize(f) > 0 &&
                     !startswith(basename(f), "."),
                readdir(dir; join = true)))
end

function load_and_downsample(dir::String, cols::Vector{Int}; sub::Int = SUB)
    files = trial_files(dir)
    R     = length(files)
    isempty(files) && error("No files found in $dir")

    # Peek at first file to get sample count
    first_trial = readdlm(files[1], Float64)
    n_raw       = size(first_trial, 1)
    n_ds        = length(1:sub:n_raw)

    @info "  $(basename(dir)): R=$R trials, n_raw=$n_raw → n_ds=$n_ds, channels=$(cols)"

    # Pre-allocate: Dict channel → Matrix(n_ds × R)
    data = Dict(c => Matrix{Float32}(undef, n_ds, R) for c in cols)

    done = Threads.Atomic{Int}(0)
    Threads.@threads for i in 1:R
        trial = readdlm(files[i], Float64)
        for c in cols
            data[c][:, i] = Float32.(trial[1:sub:end, c])
        end
        n = Threads.atomic_add!(done, 1) + 1
        n % 500 == 0 && (print("\r    $n/$R"); flush(stdout))
    end
    println("\r    $R/$R ✓")
    data
end

# ── Set 1 ─────────────────────────────────────────────────────────────────────
println("\nProcessing Set 1 (channels 1=B1, 5=B3, 7=B4)...")
isdir(set1_dir) || error("Set 1 directory not found: $set1_dir")
set1 = load_and_downsample(set1_dir, [1, 5, 7])
outpath = "data/nasa_set1.jls"
serialize(outpath, set1)
sz = filesize(outpath) / 1024^2
@printf("  Saved %s (%.1f MB)\n", outpath, sz)

# ── Set 2 ─────────────────────────────────────────────────────────────────────
println("\nProcessing Set 2 (channel 1=B1)...")
isdir(set2_dir) || error("Set 2 directory not found: $set2_dir")
set2 = load_and_downsample(set2_dir, [1])
outpath = "data/nasa_set2.jls"
serialize(outpath, set2)
sz = filesize(outpath) / 1024^2
@printf("  Saved %s (%.1f MB)\n", outpath, sz)

# ── Set 3 ─────────────────────────────────────────────────────────────────────
println("\nProcessing Set 3 (channels 1=B1, 3=B3)...")
if isdir(set3_dir)
    set3 = load_and_downsample(set3_dir, [1, 3])
    outpath = "data/nasa_set3.jls"
    serialize(outpath, set3)
    sz = filesize(outpath) / 1024^2
    @printf("  Saved %s (%.1f MB)\n", outpath, sz)
else
    @warn "Set 3 directory not found ($set3_dir) — skipping."
    @warn "Remember: Set 3 raw data is in 4th_test/txt/ — see README.md."
end

println("\nDone. Files written to data/")
println("Total size: $(round(sum(filesize(f) for f in readdir("data"; join=true)
                                if endswith(f, ".jls")) / 1024^2, digits=1)) MB")
println("\nThese files can be committed to the replication repository.")
println("empirical_topots_demo.jl will use them automatically.")
