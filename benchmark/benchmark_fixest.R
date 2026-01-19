#!/usr/bin/env Rscript
#
# Benchmark fixest (R) for comparison with Julia packages
#
# Usage: Rscript benchmark_fixest.R [nthreads] [output_path]
#

library(fixest)
library(microbenchmark)
library(data.table)

args <- commandArgs(trailingOnly = TRUE)
nthreads <- if (length(args) > 0) as.integer(args[1]) else 1
output_path <- if (length(args) > 1) args[2] else "fixest_results.csv"

cat("======================================================================\n")
cat("fixest Benchmark\n")
cat(sprintf("R version: %s\n", R.version.string))
cat(sprintf("fixest version: %s\n", packageVersion("fixest")))
cat(sprintf("Threads: %d\n", nthreads))
cat("======================================================================\n\n")

setFixest_nthreads(nthreads)

# Results storage
results <- data.frame(
    scenario = character(),
    n = integer(),
    time_seconds = numeric(),
    stringsAsFactors = FALSE
)

run_benchmark <- function(name, expr, n_obs) {
    cat(sprintf("  Running: %s\n", name))
    # Warmup
    eval(expr)
    # Benchmark
    mb <- microbenchmark(eval(expr), times = 5)
    time_sec <- median(mb$time) / 1e9
    cat(sprintf("    Time: %.3f s\n", time_sec))
    return(data.frame(scenario = name, n = n_obs, time_seconds = time_sec))
}

# ------------------------------------------------------------------------------
# Scenario 1-5: Large N dataset (10M observations)
# ------------------------------------------------------------------------------
cat("\n======================================================================\n")
cat("Creating dataset 1: 10M observations\n")
cat("======================================================================\n")

set.seed(42)
N <- 10000000
K <- 100
id1 <- sample(1:(N/K), N, replace = TRUE)
id2 <- sample(1:K, N, replace = TRUE)
x1 <- 5 * cos(id1) + 5 * sin(id2) + rnorm(N)
x2 <- cos(id1) + sin(id2) + rnorm(N)
y <- 3 * x1 + 5 * x2 + cos(id1) + cos(id2)^2 + rnorm(N)
df1 <- data.table(id1 = id1, id2 = id2, x1 = x1, x2 = x2, y = y)

cat("  Dataset created\n\n")

# Scenario 1: Simple OLS
results <- rbind(results, run_benchmark(
    "OLS: y ~ x1 + x2",
    quote(feols(y ~ x1 + x2, data = df1)),
    N
))

# Scenario 2: OLS + cluster
results <- rbind(results, run_benchmark(
    "OLS + cluster(id2)",
    quote(feols(y ~ x1 + x2, data = df1, cluster = ~id2)),
    N
))

# Scenario 3: One FE
results <- rbind(results, run_benchmark(
    "FE: y ~ x1 + x2 | id1",
    quote(feols(y ~ x1 + x2 | id1, data = df1)),
    N
))

# Scenario 4: One FE + cluster
results <- rbind(results, run_benchmark(
    "FE + cluster: | id1",
    quote(feols(y ~ x1 + x2 | id1, data = df1, cluster = ~id1)),
    N
))

# Scenario 5: Two FEs
results <- rbind(results, run_benchmark(
    "FE: y ~ x1 + x2 | id1 + id2",
    quote(feols(y ~ x1 + x2 | id1 + id2, data = df1)),
    N
))

# Clean up
rm(df1, id1, id2, x1, x2, y)
gc()

# ------------------------------------------------------------------------------
# Scenario 6: Worker-Firm structure (800K observations)
# ------------------------------------------------------------------------------
cat("\n======================================================================\n")
cat("Creating dataset 2: Worker-Firm (800K observations)\n")
cat("======================================================================\n")

set.seed(42)
N <- 800000
M <- 40000  # workers
O <- 5000   # firms
id1 <- sample(1:M, N, replace = TRUE)
id2 <- sapply(id1, function(x) sample(max(1, x %/% 8 - 10):min(O, x %/% 8 + 10), 1))
x1 <- 5 * cos(id1) + 5 * sin(id2) + rnorm(N)
x2 <- cos(id1) + sin(id2) + rnorm(N)
y <- 3 * x1 + 5 * x2 + cos(id1) + cos(id2)^2 + rnorm(N)
df2 <- data.table(id1 = id1, id2 = id2, x1 = x1, x2 = x2, y = y)

cat("  Dataset created\n\n")

# Scenario 6: Worker-Firm FE
results <- rbind(results, run_benchmark(
    "Worker-Firm FE",
    quote(feols(y ~ x1 + x2 | id1 + id2, data = df2)),
    N
))

rm(df2, id1, id2, x1, x2, y)
gc()

# ------------------------------------------------------------------------------
# Scenario 7-9: fixest-style benchmarks (10M observations, 3 FEs)
# ------------------------------------------------------------------------------
cat("\n======================================================================\n")
cat("Creating dataset 3: fixest-style (10M observations)\n")
cat("======================================================================\n")

set.seed(9876)
n <- 10000000
nb_dum <- c(n %/% 20, floor(sqrt(n)), floor(n^0.33))
id1 <- sample(1:nb_dum[1], n, replace = TRUE)
id2 <- sample(1:nb_dum[2], n, replace = TRUE)
id3 <- sample(1:nb_dum[3], n, replace = TRUE)
X1 <- runif(n)
ln_y <- 3 * X1 + runif(n)
df3 <- data.table(X1 = X1, ln_y = ln_y, id1 = id1, id2 = id2, id3 = id3)

cat("  Dataset created\n\n")

# Scenario 7: 1 FE + cluster
results <- rbind(results, run_benchmark(
    "1 FE + cluster",
    quote(feols(ln_y ~ X1 | id1, data = df3, cluster = ~id1)),
    n
))

# Scenario 8: 2 FE + cluster
results <- rbind(results, run_benchmark(
    "2 FE + cluster",
    quote(feols(ln_y ~ X1 | id1 + id2, data = df3, cluster = ~id1)),
    n
))

# Scenario 9: 3 FE + cluster
results <- rbind(results, run_benchmark(
    "3 FE + cluster",
    quote(feols(ln_y ~ X1 | id1 + id2 + id3, data = df3, cluster = ~id1)),
    n
))

# ------------------------------------------------------------------------------
# Save results
# ------------------------------------------------------------------------------
cat("\n======================================================================\n")
cat(sprintf("Saving results to %s\n", output_path))
cat("======================================================================\n")

write.csv(results, output_path, row.names = FALSE)
print(results)

cat("\nDone!\n")
