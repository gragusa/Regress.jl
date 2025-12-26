# Kleibergen-Paap F-statistic validation against Julia Regress.jl
# Run this script in R to generate reference values

library(AER)
library(sandwich)
library(lmtest)

# Set seed for reproducibility
set.seed(42)

n <- 500

# Generate data
z1 <- rnorm(n)
z2 <- rnorm(n)
e <- rnorm(n)
u <- 0.5 * e + rnorm(n)
x <- 0.5 * z1 + 0.3 * z2 + u
y <- 2.0 * x + 1.0 + e

df <- data.frame(y = y, x = x, z1 = z1, z2 = z2)

# Save data for Julia
write.csv(df, "kp_validation_data.csv", row.names = FALSE)

cat("=== TSLS Estimation ===\n")
m_tsls <- ivreg(y ~ x | z1 + z2, data = df)
summary(m_tsls, diagnostics = TRUE)

cat("\n=== Coefficients ===\n")
print(coef(m_tsls))

cat("\n=== Default vcov (assuming HC1-like) ===\n")
print(vcov(m_tsls))

cat("\n=== HC1 vcov ===\n")
V_hc1 <- vcovHC(m_tsls, type = "HC1")
print(V_hc1)

cat("\n=== HC3 vcov ===\n")
V_hc3 <- vcovHC(m_tsls, type = "HC3")
print(V_hc3)

cat("\n=== First-stage regression ===\n")
fs <- lm(x ~ z1 + z2, data = df)
summary(fs)

cat("\n=== First-stage F-statistic (standard) ===\n")
# Extract F-stat for excluded instruments
fs_summary <- summary(fs)
print(fs_summary$fstatistic)

# Robust first-stage F
cat("\n=== Robust first-stage F (Wald test with HC1) ===\n")
# Test that z1 and z2 coefficients are jointly zero
waldtest(fs, vcov = vcovHC(fs, type = "HC1"), test = "F")

cat("\n=== LIML Estimation ===\n")
m_liml <- ivreg(y ~ x | z1 + z2, data = df, method = "LIML")
summary(m_liml, diagnostics = TRUE)

cat("\n=== LIML Coefficients ===\n")
print(coef(m_liml))

cat("\n=== Fuller(1) Estimation ===\n")
# Fuller with a=1
m_fuller <- ivreg(y ~ x | z1 + z2, data = df, method = "Fuller")
summary(m_fuller, diagnostics = TRUE)

cat("\n=== Fuller Coefficients ===\n")
print(coef(m_fuller))

cat("\n\n========================================\n")
cat("SUMMARY OF KEY VALUES FOR VALIDATION\n")
cat("========================================\n")
cat("\nTSLS coefficients:\n")
print(coef(m_tsls))
cat("\nLIML coefficients:\n")
print(coef(m_liml))
cat("\nFuller(1) coefficients:\n")
print(coef(m_fuller))
cat("\nTSLS HC0 vcov:\n")
print(vcovHC(m_tsls, type = "HC0"))
cat("\nTSLS HC1 vcov:\n")
print(vcovHC(m_tsls, type = "HC1"))
cat("\nTSLS HC3 vcov:\n")
print(vcovHC(m_tsls, type = "HC3"))

cat("\nTSLS HC0 standard errors:\n")
print(sqrt(diag(vcovHC(m_tsls, type = "HC0"))))
cat("\nTSLS HC1 standard errors:\n")
print(sqrt(diag(vcovHC(m_tsls, type = "HC1"))))
cat("\nTSLS HC2 standard errors:\n")
print(sqrt(diag(vcovHC(m_tsls, type = "HC2"))))
