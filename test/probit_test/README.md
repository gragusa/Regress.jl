# Probit comparison against R fixest

This folder contains a self-contained comparison between:

- R `fixest::feglm(..., family = binomial(link = "probit"))`
- Julia `Regress.probit(...)`

Run the full comparison from the repository root:

```bash
julia --project=. test/probit_test/test_probit_vs_fixest.jl
```

Run the two regressions separately:

```bash
Rscript test/probit_test/fixest_probit.R
julia --project=. test/probit_test/regress_probit.jl
```

Optional environment variables for the Julia test:

```bash
PROBIT_BENCH_SAMPLES=10 PROBIT_FIXEST_THREADS=1 julia --project=. test/probit_test/test_probit_vs_fixest.jl
```

The test compares coefficients with a numerical tolerance and records median/minimum elapsed time for both estimators. Timing is reported but not used as a hard performance assertion.

## Latest local result

Command:

```bash
julia --project=. test/probit_test/test_probit_vs_fixest.jl
```

Result:

```text
Test Summary:                  | Pass  Total  Time
probit: Regress.jl vs R fixest |    8      8  8.5s
```

Timing comparison from that run:

| Estimator | Median seconds |
| --- | ---: |
| R fixest | 0.035 |
| Regress.jl | 0.2078 |

`Regress.jl / fixest = 5.94` on this local run.

Coefficient comparison:

| Coefficient | fixest | Regress.jl |
| --- | ---: | ---: |
| hhninc | -2.028546e-6 | -2.028526e-6 |
| hhkids | -0.0300173 | -0.0300166 |
| educ | -0.0697462 | -0.0697465 |
| married | -0.0295630 | -0.0295618 |

Note: `fixest` drops `age` because it is collinear after absorbing the fixed effects. It also reports that 3,960 fixed effects / 11,014 observations are removed because of only 0/1 outcomes or singletons. The test therefore compares the coefficient subset common to both estimators.
