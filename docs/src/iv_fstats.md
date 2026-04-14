# IV First-Stage and Weak Instrument Diagnostics

## Setup

```julia
using Regress
m = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ z1 + z2)))
```

---

## 1. IID First-Stage F (`first_stage_f`)

Standard F-test from the first-stage OLS under homoskedasticity. Equation-by-equation.

**Notation.** Let $Z_{\text{full}} = [X_{\text{exo}},\; Z]$ be the $n \times (p + \ell)$ first-stage design matrix, where $X_{\text{exo}}$ is $n \times p$ (exogenous regressors including intercept) and $Z$ is $n \times \ell$ (excluded instruments). For the $j$-th endogenous variable $x_j$:

**Computation.**

1. Restricted regression (instruments excluded):
$$\hat{e}^R_j = x_j - X_{\text{exo}}(X_{\text{exo}}'X_{\text{exo}})^{-1}X_{\text{exo}}'x_j, \qquad \text{SSR}_R = \hat{e}_j^{R\prime}\hat{e}_j^R$$

2. Full regression (instruments included):
$$\hat{e}^F_j = x_j - Z_{\text{full}}(Z_{\text{full}}'Z_{\text{full}})^{-1}Z_{\text{full}}'x_j, \qquad \text{SSR}_F = \hat{e}_j^{F\prime}\hat{e}_j^F$$

3. F-statistic:
$$F_j = \frac{(\text{SSR}_R - \text{SSR}_F)\;/\;\ell}{\text{SSR}_F\;/\;(n - p - \ell - d_{\text{fe}})}, \qquad p\text{-value from } F(\ell,\; n - p - \ell - d_{\text{fe}})$$

**Access.**

```julia
fs = first_stage_f(m)
fs.F_per_endo    # Vector, one F per endogenous variable
fs.p_per_endo    # p-values
fs.df1, fs.df2   # (ℓ, n − p − ℓ − d_fe)
```

---

## 2. Robust First-Stage F (`first_stage`)

Wald test of $H_0\!: \pi_j = 0$ in the first-stage OLS, with sandwich variance. Equation-by-equation.

**Notation.** Same $Z_{\text{full}}$, $p$, $\ell$ as above. The first-stage OLS for endogenous variable $j$ is:

$$x_j = Z_{\text{full}}\,\beta_j + e_j$$

**Computation.**

1. Coefficients (computed once for all $j$ via shared Cholesky):
$$\hat{\beta}_j = (Z_{\text{full}}'Z_{\text{full}})^{-1}Z_{\text{full}}'x_j$$

2. Fitted values and residuals:
$$\hat{\mu}_j = Z_{\text{full}}\hat{\beta}_j, \qquad \hat{e}_j = x_j - \hat{\mu}_j$$

3. Sandwich variance via CovarianceMatrices.jl — wraps $(\hat{e}_j, Z_{\text{full}}, \hat{\mu}_j)$ in a lightweight `OLSMatrixEstimator` and calls:
$$V_j = \text{vcov}(\texttt{vcov\_type},\;\text{model}_j)$$
   This handles HC0–HC5, CR0–CR3, HAC, EWC, etc.

4. Extract the instrument-coefficient block and Wald test:
$$V_{\pi_j} = V_j[p\!+\!1:p\!+\!\ell,\; p\!+\!1:p\!+\!\ell], \qquad \hat{\pi}_j = \hat{\beta}_j[p\!+\!1:p\!+\!\ell]$$

$$F_j = \frac{\hat{\pi}_j'\,V_{\pi_j}^{-1}\,\hat{\pi}_j}{\ell}, \qquad p\text{-value from } F(\ell,\; n - p - \ell - d_{\text{fe}})$$

**Access.**

```julia
fs = first_stage(m)
fs.F_nonrobust   # IID F (same as first_stage_f)
fs.F_robust      # Robust Wald F (uses model's vcov, default HC1)
fs.vcov_type     # String name of vcov estimator

# Switch variance estimator without refitting:
m_hc3 = m + vcov(HC3())
first_stage(m_hc3).F_robust     # recomputed with HC3
first_stage(m_hc3).F_nonrobust  # unchanged
```

---

## 3. Montiel-Olea & Pflueger Weak IV Test (`weakivtest`)

Tests for weak instruments. Single endogenous regressor only. Returns three F-statistics and critical values.

**Notation.** Let $n$ observations, $\ell$ excluded instruments, $p$ exogenous regressors.

**Computation.**

1. Partial out exogenous regressors via QR:
$$\tilde{y} = M_{X_{\text{exo}}}\,y, \qquad \tilde{x} = M_{X_{\text{exo}}}\,x_{\text{endo}}$$
   where $M_A = I - A(A'A)^{-1}A'$.

2. Orthogonalize instruments: $Z_s = Q\sqrt{n}$ from $\text{QR}(M_{X_{\text{exo}}}Z)$, so $Z_s'Z_s = nI$.

3. Reduced-form coefficients:
$$\hat{\pi} = Z_s'\tilde{x}/n, \qquad \hat{d} = Z_s'\tilde{y}/n$$

4. Reduced-form residuals:
$$\hat{u}_x = \tilde{x} - Z_s\hat{\pi}, \qquad \hat{u}_y = \tilde{y} - Z_s\hat{d}$$

5. Residual covariance ($\delta = n - \ell - p$):
$$\Omega = \begin{pmatrix}\hat{u}_y'\hat{u}_y & \hat{u}_y'\hat{u}_x \\ \hat{u}_x'\hat{u}_y & \hat{u}_x'\hat{u}_x\end{pmatrix} / \delta$$

6. Sandwich variance of stacked moments $M = [\hat{u}_y \odot Z_s,\;\hat{u}_x \odot Z_s]$ (size $n \times 2\ell$):
$$W = \frac{\text{meat}(M,\;\texttt{vcov\_type})}{\delta \cdot c_{\text{clust}}}$$
   where $c_{\text{clust}}$ is 1 for HC, or cluster DOF adjustment for CR. Extract:
$$W_2 = W[\ell\!+\!1:2\ell,\;\ell\!+\!1:2\ell]$$

7. Three F-statistics:

$$F_{\text{nonrobust}} = \frac{n\,\hat{\pi}'\hat{\pi}}{\ell\,\omega_{22}}, \qquad \omega_{22} = \Omega[2,2]$$

$$F_{\text{eff}} = \frac{n\,\hat{\pi}'\hat{\pi}}{\text{tr}(W_2)}$$

$$F_{\text{robust}} = \frac{n\,\hat{\pi}'\,W_2^{-1}\,\hat{\pi}}{\ell}$$

**Interpretation.** Compare $F_{\text{eff}}$ to `cv_TSLS`/`cv_LIML` and $F_{\text{robust}}$ to `cv_GMMf`. If $F > \text{cv}[\tau]$, worst-case bias is below $\tau$ of the OLS bias.

**Access.**

```julia
r = weakivtest(m)
r.F_eff, r.F_robust, r.F_nonrobust

# Critical values at τ ∈ {5%, 10%, 20%, 30%}:
r.cv_TSLS    # compare F_eff
r.cv_LIML    # compare F_eff
r.cv_GMMf    # compare F_robust

# Estimator coefficients and SEs:
r.btsls, r.sebtsls   # TSLS
r.bliml, r.sebliml   # LIML
r.bgmmf, r.sebgmmf   # GMMf
r.kappa              # LIML kappa
```

---

---

## 4. Kleibergen-Paap rk Wald Statistic (internal)

Joint rank test for under-identification. Computed at fitting time via `ranktest()` but **not currently stored** on the model struct — only the per-endogenous Wald F-statistics (Section 2) are exposed.

**Notation.** Let $\hat\Pi$ be the $\ell \times k$ matrix of first-stage coefficients on excluded instruments (after partialling out exogenous regressors). Let $Z_{\text{res}}$ ($n \times \ell$) and $X_{\text{endo,res}}$ ($n \times k$) be residualized instruments and endogenous variables.

**Computation.**

1. Cholesky factorizations:
$$F F' = Z_{\text{res}}'Z_{\text{res}}, \qquad G G' = X_{\text{endo,res}}'X_{\text{endo,res}}$$

2. Normalized coefficient matrix and its SVD:
$$\Theta = F\,\hat\Pi'\,(G')^{-1}, \qquad \Theta = U\,S\,V'$$

3. Extract test direction from SVD submatrices at rank $k$:
$$\lambda = (b_{kk} \otimes a_{kk}')\,\text{vec}(\Theta)$$

4. Variance of $\lambda$:
   - **IID:** $V_\lambda = (\text{kron}_v\,\text{kron}_v')\;/\;n$
   - **Robust:** $V_\lambda = \text{kron}_v\;\hat{V}\;\text{kron}_v'$, where $\hat{V}$ is the sandwich variance of $\text{vec}(\hat\Pi)$ computed from the moment matrix $Z_{\text{res}} \odot \hat{e}$.

5. Test statistic:
$$r_{KP} = \lambda'\,V_\lambda^{-1}\,\lambda, \qquad F_{KP} = r_{KP}\;/\;\ell$$

Under $H_0$ (rank deficiency), $r_{KP} \sim \chi^2(\ell - k + 1)$.

**Access.**

```julia
r = first_stage_F_KP(m)
r.stat      # F = r_KP / ℓ
r.df1       # ℓ (number of excluded instruments)
r.p         # p-value from χ²(ℓ − k + 1)
```

---

## 5. Wu-Hausman Endogeneity Test (`wu_hausman`)

Tests $H_0$: the instrumented variables are exogenous (OLS is consistent). Rejection suggests endogeneity and justifies IV estimation.

**Computation.** Let $X = [X_{\text{exo}},\; X_{\text{endo}}]$ be the second-stage regressors and $\hat{v}$ the first-stage residuals ($n \times k$ matrix, from regressing each endogenous variable on all instruments).

1. Restricted model (OLS, no endogeneity correction):
$$\hat{e}_R = y - X(X'X)^{-1}X'y, \qquad \text{SSR}_R = \hat{e}_R'\hat{e}_R$$

2. Unrestricted model (augmented with first-stage residuals):
$$W = [X,\;\hat{v}], \qquad \hat{e}_U = y - W(W'W)^{-1}W'y, \qquad \text{SSR}_U = \hat{e}_U'\hat{e}_U$$

3. F-statistic:
$$F = \frac{(\text{SSR}_R - \text{SSR}_U)\;/\;k}{\text{SSR}_U\;/\;(d_{\text{res}} - k)}, \qquad p\text{-value from } F(k,\; d_{\text{res}} - k)$$

where $k$ is the number of endogenous variables and $d_{\text{res}}$ is the second-stage residual degrees of freedom.

**Access.**

```julia
r = wu_hausman(m)
r.stat      # F-statistic
r.p         # p-value
r.df1       # k (number of endogenous variables)
r.df2       # d_res − k
```

---

## 6. Sargan Overidentification Test (`sargan`)

Tests $H_0$: the excluded instruments are valid (uncorrelated with the structural error). Only meaningful when the model is overidentified ($\ell > k$).

**Computation.** Let $e = y - X\hat\beta_{\text{IV}}$ be the second-stage residuals and $Z_{\text{full}} = [X_{\text{exo}},\; Z]$ the full instrument matrix.

1. Auxiliary regression of residuals on all instruments:
$$\hat\gamma = (Z_{\text{full}}'Z_{\text{full}})^{-1}Z_{\text{full}}'e, \qquad \hat{e}_{\text{aux}} = Z_{\text{full}}\,\hat\gamma$$

2. $R^2$ from auxiliary regression:
$$R^2 = 1 - \frac{\|e - \hat{e}_{\text{aux}}\|^2}{\|e\|^2}$$

3. Test statistic:
$$S = n \cdot R^2 \;\sim\; \chi^2(\ell - k)$$

where $\ell$ is the number of excluded instruments and $k$ the number of endogenous variables.

**Access.**

```julia
r = sargan(m)
r.stat      # n × R² (χ² statistic)
r.p         # p-value from χ²(ℓ − k)
r.df        # ℓ − k
```

---

## Summary

### First-Stage Diagnostics

| Statistic | Access | Variance | Scope |
|-----------|--------|----------|-------|
| IID F | `first_stage_F_iid(m).stat` | Homoskedastic | Per endogenous |
| Robust Wald F | `first_stage_F_robust(m).stat` | Model's vcov | Per endogenous |
| KP rk Wald F | `first_stage_F_KP(m).stat` | Model's vcov | Joint |

### Weak Instrument Test

| Statistic | Access | Variance | Compare to |
|-----------|--------|----------|------------|
| Effective F (MOP) | `weakivtest(m).F_eff` | Model's vcov | `cv_TSLS`, `cv_LIML` |
| Robust F (Windmeijer) | `weakivtest(m).F_robust` | Model's vcov | `cv_GMMf` |
| Non-robust F | `weakivtest(m).F_nonrobust` | Homoskedastic | — |

### Specification Tests

| Test | Access | $H_0$ | Distribution |
|------|--------|-------|--------------|
| Wu-Hausman | `wu_hausman(m)` | Endogenous vars are exogenous | $F(k, d_{\text{res}} - k)$ |
| Sargan | `sargan(m)` | Instruments are valid | $\chi^2(\ell - k)$ |

**References.**
Montiel Olea & Pflueger (2013), *JBES*.
Windmeijer (2025), *Journal of Econometrics*.
Sargan (1958), *Econometrica*.
Wu (1973), *Econometrica*; Hausman (1978), *Econometrica*.
