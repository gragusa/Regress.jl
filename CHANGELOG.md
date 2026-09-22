# Changelog

## Unreleased

## 0.2.0

### Breaking Changes

- **Requires CovarianceMatrices 0.32.** Earlier versions no longer load: 0.32
  removed `setkernelweights!`, on which Regress defined methods.

- **Requires Julia 1.12.** CovarianceMatrices 0.32 sets this floor, which rules
  out the 1.10 long-term-support release.

- **`vcov(model)` returns a `CovarianceMatrix`** instead of
  `Symmetric{T,Matrix{T}}`. It is still an `AbstractMatrix`, so indexing,
  `inv`, `\`, and factorizations are unchanged, but the result now carries the
  metadata of the estimate: `estimator(V)` gives the variance estimator,
  `bandwidth(V)` the selected HAC bandwidth, `kernelweights(V)` the per-column
  kernel weights, `information(V)` the full metadata `NamedTuple`, and
  `parent(V)` the bare matrix. Code that stores the result in a
  `Matrix`- or `Symmetric`-typed field needs updating; code that only reads
  values does not.

- **`weakivtest` takes `tol` instead of `eps`.** The old name shadowed
  `Base.eps` in the method body. `weakivtest(m; eps = ...)` now errors.

- **`IVMatrixEstimator` carries the estimator it was fitted with.** The type is
  now `IVMatrixEstimator{T, E, V, C}`, with a leading `estimator::E` field, so
  that leverage and display can dispatch on it. Code that writes the type with
  all its parameters, or constructs one positionally, needs updating;
  `M <: IVMatrixEstimator` bounds and field access by name are unaffected.

- **`LagTerm` is public rather than exported.** It names the type behind
  `lags(...)` and is not meant to be constructed directly, so `using Regress`
  no longer brings it into scope. Use `Regress.LagTerm` where you need the
  name; `lags` itself is unchanged.

- **Invalid input raises `ArgumentError`.** Validation failures previously
  threw bare strings, which `throw` wraps in `ErrorException`. Messages are
  unchanged, but `catch`/`@test_throws` clauses matching on the exception type
  need updating.

- **The matrix API rejects offset arrays.** `ols(X, y)` and
  `iv(::TSLS, Z, X, y)` index positionally after converting their arguments,
  so they now declare `Base.require_one_based_indexing`. `view`s and ordinary
  arrays are unaffected; an `OffsetArray` argument errors at the entry point
  instead of producing a misaligned fit.

### New Features

- **`lags()` formula term**: `@formula(y ~ lags(x, 12))` expands into a matrix of 12 lag columns. Supports nested transforms (`lags(log(abs(x)), 3)`), interactions (`lags(x, 3) & z`), and composition with other terms. Moved from LocalProjections.jl so both packages share the same implementation.

- **`first_stage_F_iid(m)`**, **`first_stage_F_robust(m)`**, **`first_stage_F_KP(m)`**: New first-stage diagnostic API returning `FirstStageFTest{T, K}` — parametric on value type (`Float64` or `Vector{Float64}`) and variance estimator type. Replaces the ambiguous `first_stage(m)` / `first_stage_f(m)` functions.

- **Kleibergen-Paap rk Wald F-statistic exposed**: The joint KP test statistic is now stored on the model and accessible via `first_stage_F_KP(m)`. Previously computed but discarded.

- **`AbstractTest` type hierarchy**: `AbstractTest` is the new abstract supertype for `FirstStageFTest{T, K}`, `WeakIVTestResult{T}`, `FirstStageResult`, `FirstStageIV`, `WuHausmanResult` and `SarganResult`. Those with a single well-defined p-value implement `StatsAPI.pvalue`; `SarganResult` also implements `dof`. `WeakIVTestResult` implements neither, since it reports against critical values.

- **Public model surface**: `OLSEstimator`, `IVEstimator`, `OLSMatrixEstimator`, `IVMatrixEstimator`, `AbstractIVEstimator`, `esample` and `partial_out` are marked `public`, so `Regress.OLSEstimator` and friends resolve without being exported.

- **`Homoskedastic` type**: Sentinel type used as the variance estimator parameter for IID F-tests.

- **NaN row filtering for `lags()`**: Rows with NaN values produced by `lags()` in the design matrix are automatically excluded from estimation in OLS, TSLS, and K-class models. The `esample` field correctly reflects these exclusions.

- **EWC, DriscollKraay, VARHAC support**: All `Correlated` variance estimators from CovarianceMatrices.jl now work with `model + vcov(...)` (previously only `HAC` was supported).

- **Driscoll-Kraay standard errors on fitted models**: `model + vcov(DriscollKraay(...))`
  now works for OLS and IV models and for the matrix estimators. Previously
  every such call was a `MethodError`. The computation delegates to
  CovarianceMatrices, which scales by the number of time periods rather than
  the number of observations.

- **`show` reports the selected HAC bandwidth**: a model fitted with an
  automatic bandwidth selector prints `Bartlett(auto: 55.71)` rather than
  `Bartlett(auto)`. The bandwidth is a property of the estimate, so it is read
  from `vcov(model)`.

### Bug Fixes

- **Over-identified TSLS from matrix inputs used the wrong leverage.**
  `leverage(::IVMatrixEstimator)` computed `diag(X̂(X̂'X̂)⁻¹X̂')`, which equals
  the AER/sandwich IV leverage `diag(X·(X̂'X̂)⁻¹·X'·Z·(Z'Z)⁻¹·Z')` only when the
  model is just-identified. HC2/HC3 standard errors from `iv(TSLS(), Z, X, y)`
  therefore differed from those of the same model fitted through the formula
  interface. Both paths now agree; HC0/HC1 and just-identified models were
  never affected.

- **CR2 uses the symmetric square root of `I - H_gg`.** The Bell-McCaffrey
  adjustment is defined by the symmetric root; the Cholesky factor used
  previously preserves the quadratic form `u'(I - H_gg)⁻¹u` but differs from it
  by an orthogonal rotation, which does not cancel in the outer products the
  cluster meat sums. CR2 standard errors change by a fraction of a percent on
  typical data. CR3 is unaffected.

- **CR2/CR3 on weighted models no longer double-count the weights.**
  `modelmatrix` and `residuals` already carry the weighting, so the per-cluster
  leverage blocks must not be weighted again. They were, which drove
  `I - H_gg` indefinite and produced standard errors roughly an order of
  magnitude too large (and, for CR2, an `InexactError` on some data).

- **CR2/CR3 work on IV models.** Both threw `MethodError` when resolving cluster
  indices. The leverage blocks are built from the second-stage regressor matrix,
  which is what the IV sandwich treats as the design.

- **HC4/HC5 on the matrix IV estimator use the standard exponents.**
  `IVMatrixEstimator` raised `1 - h` to `δ` where the definition uses `δ/2`, and
  took an extra square root for HC5, so its adjustments disagreed with the ones
  the other three model types compute.

- **HAC bandwidth selection accounts for kernel weights.** The kernel weights
  that give the intercept column zero weight are now passed to `aVar` through
  the `weights` keyword. The moment matrix's intercept column is not constant,
  so an Andrews or Newey-West bandwidth computed without them differs — by a
  factor of about two on `y ~ x` in the bundled validation data.

- **CovarianceMatrices entry points accept Regress models.** `bread` and
  `leverage` were defined as new functions in Regress's namespace instead of
  extending the CovarianceMatrices generics, so `CovarianceMatrices.vcov(k, m)`
  threw a `MethodError` for every Regress model. `numobs` and `mask` were
  missing for the IV estimators. All are now provided, and the HC2-HC5 and
  CR2/CR3 estimators that dispatch on `leverage` work through the upstream path.

- **`aVar(k, model)` no longer masks collinear coefficients.** `aVar` estimates
  the variance of the moment matrix, where coefficient collinearity has not yet
  entered; rank deficiency is handled by `vcov`, which subsets to the
  full-rank block independently. The masking discarded the `CovarianceMatrix`
  wrapper without affecting any variance. `vcov` results are unchanged,
  including the NaN entries it produces for collinear coefficients.

- **HC3 first-stage F-statistic**: The robust first-stage F with HC3 (and HC2/HC4/HC5) was silently using the HC1 formula. Refactored to delegate to CovarianceMatrices.jl, which handles all variance types correctly.

- **`esample` with `lags()`**: `esample` now correctly marks NaN-filtered rows as `false`, so `sum(m.esample) == nobs(m)` and residuals can be mapped back to the original DataFrame via `res[m.esample] .= residuals(m)`.

- **`lags()` with nested transforms and other terms**: `@formula(y ~ lags(log(abs(x)), 3) + log(x))` previously crashed because `StatsModels.terms(::LagTerm)` returned a `FunctionTerm` instead of leaf terms.

- **`lags()` coefficient names for nested transforms**: `lags(log(abs(x)), 3)` produced names like `l_lag1` instead of `log(abs(x))_lag1`.

- **`_compute_meat` cluster aggregation**: The Kleibergen-Paap rank test's cluster-robust meat computation treated a `Clustering` struct as a raw vector. Fixed to use `.groups` / `.ngroups`.

- **Fixest validation F-stat references**: Test reference values for `F_nonrobust` were Kleibergen-Paap statistics, not IID F-statistics. Corrected to match the actual SSR-based F-test.

- **`first_stage with vcov` test**: HC1 and HC3 first-stage robust F-statistics were identical due to the HC3 fallthrough bug.

- **Fallback path in `compute_per_endogenous_fstats`**: The fallback when `Xendo_orig` is `nothing` passed `Z_res` as the full design matrix, producing wrong F-statistics. Replaced with an explicit error.

### Refactoring

- **Residual adjustments delegate to CovarianceMatrices.jl**: the HC0 and HC2-HC5
  adjustments, and the CR2/CR3 leverage blocks, now come from
  `CovarianceMatrices.residual_adjustment` instead of being reimplemented for each
  of the four model types. HC1 stays local because upstream divides by
  `n - length(coef(m))`, which cannot see fixed effects absorbed out of the
  design, and the CR finite-sample corrections stay local to keep matching fixest.

- **First-stage robust F via CovarianceMatrices.jl**: Replaced ~250 lines of manual sandwich variance computation (`_compute_meat_inplace!`, `_compute_robust_first_stage_fstats_batched`, `_compute_single_first_stage_fstat`) with `_compute_first_stage_fstats_via_ols`, which constructs lightweight `OLSMatrixEstimator` wrappers around pre-computed first-stage data and delegates to `CovarianceMatrices.vcov`. No refitting; ZZ factorization is shared across endogenous variables.

- **`_filter_nan_rows` shared utility**: Extracted the NaN-filtering logic (previously triplicated in `fit_ols.jl`, `tsls.jl`, `kclass.jl`) into a single function in `fit_common.jl`.

- **`LagTerm` multi-column guard**: `modelcols(::LagTerm)` now throws `ArgumentError` for multi-column inner terms (interactions, categoricals) instead of producing silently wrong results.

- **LocalProjections.jl integration**: `first_stage` and `weakivtest` in LocalProjections.jl now extend the Regress.jl functions instead of defining separate ones.

- **Test import cleanup**: Replaced `using CovarianceMatrices: ...` with `using Regress: ...` in test files where CovarianceMatrices is not directly loadable. Eliminated `Base.==` method redefinition warnings in `test_formula.jl`.

### Documentation

- **`docs/src/iv_fstats.md`**: New document covering all IV diagnostics — IID F, robust Wald F, Kleibergen-Paap rank test, Montiel-Olea-Pflueger weak IV test, Wu-Hausman endogeneity test, and Sargan overidentification test — with exact formulas, computation steps, and API examples.
