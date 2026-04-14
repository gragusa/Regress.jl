# Changelog

## Unreleased

### New Features

- **`lags()` formula term**: `@formula(y ~ lags(x, 12))` expands into a matrix of 12 lag columns. Supports nested transforms (`lags(log(abs(x)), 3)`), interactions (`lags(x, 3) & z`), and composition with other terms. Moved from LocalProjections.jl so both packages share the same implementation.

- **`first_stage_F_iid(m)`**, **`first_stage_F_robust(m)`**, **`first_stage_F_KP(m)`**: New first-stage diagnostic API returning `FirstStageFTest{T, K}` — parametric on value type (`Float64` or `Vector{Float64}`) and variance estimator type. Replaces the ambiguous `first_stage(m)` / `first_stage_f(m)` functions.

- **Kleibergen-Paap rk Wald F-statistic exposed**: The joint KP test statistic is now stored on the model and accessible via `first_stage_F_KP(m)`. Previously computed but discarded.

- **`AbstractTest` type hierarchy**: `AbstractTest` is the new abstract supertype for `FirstStageFTest{T, K}` and `WeakIVTestResult{T}`.

- **`Homoskedastic` type**: Sentinel type used as the variance estimator parameter for IID F-tests.

- **NaN row filtering for `lags()`**: Rows with NaN values produced by `lags()` in the design matrix are automatically excluded from estimation in OLS, TSLS, and K-class models. The `esample` field correctly reflects these exclusions.

- **EWC, DriscollKraay, VARHAC support**: All `Correlated` variance estimators from CovarianceMatrices.jl now work with `model + vcov(...)` (previously only `HAC` was supported).

### Bug Fixes

- **HC3 first-stage F-statistic**: The robust first-stage F with HC3 (and HC2/HC4/HC5) was silently using the HC1 formula. Refactored to delegate to CovarianceMatrices.jl, which handles all variance types correctly.

- **`esample` with `lags()`**: `esample` now correctly marks NaN-filtered rows as `false`, so `sum(m.esample) == nobs(m)` and residuals can be mapped back to the original DataFrame via `res[m.esample] .= residuals(m)`.

- **`lags()` with nested transforms and other terms**: `@formula(y ~ lags(log(abs(x)), 3) + log(x))` previously crashed because `StatsModels.terms(::LagTerm)` returned a `FunctionTerm` instead of leaf terms.

- **`lags()` coefficient names for nested transforms**: `lags(log(abs(x)), 3)` produced names like `l_lag1` instead of `log(abs(x))_lag1`.

- **`_compute_meat` cluster aggregation**: The Kleibergen-Paap rank test's cluster-robust meat computation treated a `Clustering` struct as a raw vector. Fixed to use `.groups` / `.ngroups`.

- **Fixest validation F-stat references**: Test reference values for `F_nonrobust` were Kleibergen-Paap statistics, not IID F-statistics. Corrected to match the actual SSR-based F-test.

- **`first_stage with vcov` test**: HC1 and HC3 first-stage robust F-statistics were identical due to the HC3 fallthrough bug.

- **Fallback path in `compute_per_endogenous_fstats`**: The fallback when `Xendo_orig` is `nothing` passed `Z_res` as the full design matrix, producing wrong F-statistics. Replaced with an explicit error.

### Refactoring

- **First-stage robust F via CovarianceMatrices.jl**: Replaced ~250 lines of manual sandwich variance computation (`_compute_meat_inplace!`, `_compute_robust_first_stage_fstats_batched`, `_compute_single_first_stage_fstat`) with `_compute_first_stage_fstats_via_ols`, which constructs lightweight `OLSMatrixEstimator` wrappers around pre-computed first-stage data and delegates to `CovarianceMatrices.vcov`. No refitting; ZZ factorization is shared across endogenous variables.

- **`_filter_nan_rows` shared utility**: Extracted the NaN-filtering logic (previously triplicated in `fit_ols.jl`, `tsls.jl`, `kclass.jl`) into a single function in `fit_common.jl`.

- **`LagTerm` multi-column guard**: `modelcols(::LagTerm)` now throws `ArgumentError` for multi-column inner terms (interactions, categoricals) instead of producing silently wrong results.

- **LocalProjections.jl integration**: `first_stage` and `weakivtest` in LocalProjections.jl now extend the Regress.jl functions instead of defining separate ones.

- **Test import cleanup**: Replaced `using CovarianceMatrices: ...` with `using Regress: ...` in test files where CovarianceMatrices is not directly loadable. Eliminated `Base.==` method redefinition warnings in `test_formula.jl`.

### Documentation

- **`docs/src/iv_fstats.md`**: New document covering all IV diagnostics — IID F, robust Wald F, Kleibergen-Paap rank test, Montiel-Olea-Pflueger weak IV test, Wu-Hausman endogeneity test, and Sargan overidentification test — with exact formulas, computation steps, and API examples.
