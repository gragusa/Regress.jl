using TestItemRunner
using Regress

# Filter by tags via environment variable
# GPU tests are skipped unless GPU_TEST=true
# Validation tests can be excluded via TI_EXCLUDE=validation
testfilter = ti -> begin
    exclude = Symbol[]
    # Skip GPU tests unless GPU_TEST=true
    if get(ENV, "GPU_TEST", "") != "true"
        push!(exclude, :gpu)
    end
    return all(!in(exclude), ti.tags)
end

println("Running Regress.jl tests with $(Threads.nthreads()) threads...")
@run_package_tests filter=testfilter
