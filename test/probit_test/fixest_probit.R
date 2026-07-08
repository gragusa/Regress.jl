#!/usr/bin/env Rscript

# Equivalent R/fixest probit benchmark for:
# visit_dummy ~ age + hhninc + hhkids + educ + married + id FE + year FE
#
# Usage:
#   Rscript test/probit_test/fixest_probit.R [output_csv] [samples] [nthreads]

library(fixest)

args <- commandArgs(trailingOnly = TRUE)

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
script_dir <- if (length(script_path) == 0) getwd() else dirname(normalizePath(script_path))
repo_root <- normalizePath(file.path(script_dir, "..", ".."))

output_csv <- if (length(args) >= 1) args[[1]] else file.path(script_dir, "fixest_probit_results.csv")
samples <- if (length(args) >= 2) as.integer(args[[2]]) else 5L
nthreads <- if (length(args) >= 3) as.integer(args[[3]]) else 1L

setFixest_nthreads(nthreads)

data_path <- file.path(repo_root, "test", "data", "rwm.data")
df <- read.table(data_path, header = FALSE)
names(df) <- c(
  "id", "female", "year", "age", "hsat", "handdum", "handper",
  "hhninc", "hhkids", "educ", "married", "haupts", "reals",
  "fachhs", "abitur", "univ", "working", "bluec", "whitec",
  "self", "beamt", "docvis", "hospvis", "public", "addon"
)

df$visit_dummy <- as.integer(df$docvis > 0)

formula_fixest <- visit_dummy ~ age + hhninc + hhkids + educ + married | id + year

# Warmup: avoid measuring package/method initialization.
model <- feglm(formula_fixest, data = df, family = binomial(link = "probit"))

times <- replicate(samples, {
  gc()
  system.time(
    feglm(formula_fixest, data = df, family = binomial(link = "probit"))
  )[["elapsed"]]
})

coefs <- coef(model)

results <- rbind(
  data.frame(kind = "coef", name = names(coefs), value = as.numeric(coefs)),
  data.frame(kind = "time", name = "median_seconds", value = median(times)),
  data.frame(kind = "time", name = "minimum_seconds", value = min(times)),
  data.frame(kind = "meta", name = "nobs", value = nobs(model)),
  data.frame(kind = "meta", name = "loglikelihood", value = as.numeric(logLik(model))),
  data.frame(kind = "meta", name = "samples", value = samples),
  data.frame(kind = "meta", name = "nthreads", value = getFixest_nthreads())
)

write.csv(results, output_csv, row.names = FALSE)
cat("fixest probit results written to:", output_csv, "\n")
