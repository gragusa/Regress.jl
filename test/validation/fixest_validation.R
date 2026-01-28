# R script for generating reference values from fixest package
# Run: Rscript fixest_validation.R
#
# Dependencies: fixest, data.table
#
# This script estimates OLS and IV models with various FE and vcov specifications,
# then saves the results for comparison with Regress.jl.

library(fixest)
library(data.table)

# Read the validation data
data_path <- file.path(dirname(sys.frame(1)$ofile), "fixest_validation_data.csv")
if (!file.exists(data_path)) {
    # Try current directory
    data_path <- "fixest_validation_data.csv"
}
df <- fread(data_path)

# Convert Z_cat to factor for categorical instrument
df$Z_cat <- as.factor(df$Z_cat)

# Results container
results <- data.table()

# Helper function to extract results from a model
extract_results <- function(model, model_name, vcov_type, vcov_label) {
    # Get coefficients
    coefs <- coef(model)

    # Get standard errors with specified vcov
    if (vcov_type == "hetero") {
        se_vals <- sqrt(diag(vcov(model, vcov = "hetero")))
    } else if (vcov_type == "cluster_fe1") {
        se_vals <- sqrt(diag(vcov(model, vcov = ~fe1)))
    } else if (vcov_type == "cluster_twoway") {
        se_vals <- sqrt(diag(vcov(model, vcov = ~fe1 + fe2)))
    } else {
        se_vals <- sqrt(diag(vcov(model)))
    }

    # Get first-stage F for IV models
    fs_F <- NA
    if (inherits(model, "fixest") && !is.null(model$iv)) {
        # Get first-stage F-stat from fitstat
        fs <- fitstat(model, "ivf", verbose = FALSE)
        if (!is.null(fs$ivf)) {
            fs_F <- fs$ivf$stat
        }
    }

    # Create results for each coefficient
    coef_names <- names(coefs)
    dt <- data.table(
        model = model_name,
        vcov_type = vcov_label,
        coef_name = coef_names,
        coef_value = as.numeric(coefs),
        se = as.numeric(se_vals),
        nobs = nobs(model),
        r2 = r2(model, "r2"),
        first_stage_F = fs_F
    )

    return(dt)
}

# Helper to run model with multiple vcov types
run_model_vcovs <- function(model, model_name) {
    rbindlist(list(
        extract_results(model, model_name, "hetero", "HC1"),
        extract_results(model, model_name, "cluster_fe1", "cluster_fe1"),
        extract_results(model, model_name, "cluster_twoway", "cluster_fe1_fe2")
    ))
}

cat("Estimating models...\n")

# ============================================================================
# OLS Models
# ============================================================================

# Model 1: OLS no FE
cat("  OLS no FE\n")
m_ols_nofe <- feols(y ~ x1 + x2 + endo, data = df)
results <- rbind(results, run_model_vcovs(m_ols_nofe, "ols_nofe"))

# Model 2: OLS + fe1
cat("  OLS + fe1\n")
m_ols_fe1 <- feols(y ~ x1 + x2 + endo | fe1, data = df)
results <- rbind(results, run_model_vcovs(m_ols_fe1, "ols_fe1"))

# Model 3: OLS + fe1 + fe2
cat("  OLS + fe1 + fe2\n")
m_ols_fe1_fe2 <- feols(y ~ x1 + x2 + endo | fe1 + fe2, data = df)
results <- rbind(results, run_model_vcovs(m_ols_fe1_fe2, "ols_fe1_fe2"))

# ============================================================================
# IV Models with Z_continuous
# ============================================================================

# Model 4: IV Z_continuous no FE
cat("  IV Z_continuous no FE\n")
m_iv_zc_nofe <- feols(y ~ x1 + x2 | endo ~ Z_continuous, data = df)
results <- rbind(results, run_model_vcovs(m_iv_zc_nofe, "iv_Zcont_nofe"))

# Model 5: IV Z_continuous + fe1
cat("  IV Z_continuous + fe1\n")
m_iv_zc_fe1 <- feols(y ~ x1 + x2 | fe1 | endo ~ Z_continuous, data = df)
results <- rbind(results, run_model_vcovs(m_iv_zc_fe1, "iv_Zcont_fe1"))

# Model 6: IV Z_continuous + fe1 + fe2
cat("  IV Z_continuous + fe1 + fe2\n")
m_iv_zc_fe1_fe2 <- feols(y ~ x1 + x2 | fe1 + fe2 | endo ~ Z_continuous, data = df)
results <- rbind(results, run_model_vcovs(m_iv_zc_fe1_fe2, "iv_Zcont_fe1_fe2"))

# ============================================================================
# IV Models with Z_cat (categorical instrument)
# ============================================================================

# Model 7: IV Z_cat no FE
cat("  IV Z_cat no FE\n")
m_iv_zcat_nofe <- feols(y ~ x1 + x2 | endo ~ Z_cat, data = df)
results <- rbind(results, run_model_vcovs(m_iv_zcat_nofe, "iv_Zcat_nofe"))

# Model 8: IV Z_cat + fe1
cat("  IV Z_cat + fe1\n")
m_iv_zcat_fe1 <- feols(y ~ x1 + x2 | fe1 | endo ~ Z_cat, data = df)
results <- rbind(results, run_model_vcovs(m_iv_zcat_fe1, "iv_Zcat_fe1"))

# Model 9: IV Z_cat + fe1 + fe2
cat("  IV Z_cat + fe1 + fe2\n")
m_iv_zcat_fe1_fe2 <- feols(y ~ x1 + x2 | fe1 + fe2 | endo ~ Z_cat, data = df)
results <- rbind(results, run_model_vcovs(m_iv_zcat_fe1_fe2, "iv_Zcat_fe1_fe2"))

# ============================================================================
# IV Models with both Z_continuous + Z_cat
# ============================================================================

# Model 10: IV both no FE
cat("  IV both no FE\n")
m_iv_both_nofe <- feols(y ~ x1 + x2 | endo ~ Z_continuous + Z_cat, data = df)
results <- rbind(results, run_model_vcovs(m_iv_both_nofe, "iv_both_nofe"))

# Model 11: IV both + fe1
cat("  IV both + fe1\n")
m_iv_both_fe1 <- feols(y ~ x1 + x2 | fe1 | endo ~ Z_continuous + Z_cat, data = df)
results <- rbind(results, run_model_vcovs(m_iv_both_fe1, "iv_both_fe1"))

# Model 12: IV both + fe1 + fe2
cat("  IV both + fe1 + fe2\n")
m_iv_both_fe1_fe2 <- feols(y ~ x1 + x2 | fe1 + fe2 | endo ~ Z_continuous + Z_cat, data = df)
results <- rbind(results, run_model_vcovs(m_iv_both_fe1_fe2, "iv_both_fe1_fe2"))

# ============================================================================
# Save results
# ============================================================================

output_path <- file.path(dirname(sys.frame(1)$ofile), "fixest_validation_summary.csv")
if (!file.exists(dirname(output_path))) {
    output_path <- "fixest_validation_summary.csv"
}
fwrite(results, output_path)

cat("\nResults saved to:", output_path, "\n")
cat("Total rows:", nrow(results), "\n")
cat("Models:", unique(results$model), "\n")
