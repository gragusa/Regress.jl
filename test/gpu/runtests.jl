using TestItemRunner
using Regress

# Run ONLY GPU tests (requires GPU hardware)
testfilter = ti -> :gpu in ti.tags

println("Running Regress.jl GPU tests with $(Threads.nthreads()) threads...")
@run_package_tests filter=testfilter
