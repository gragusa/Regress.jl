# Single-Threaded Benchmark

## Configuration

- **BLAS Library:** /Users/gragusa/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/lib/julia/libopenblas64_.dylib
- **BLAS Threads:** 1
- **Benchmark Type:** Single-Threaded (Baseline)
- **Description:** All packages running single-threaded for baseline comparison
- **Julia Threads:** 5
- **Julia Version:** 1.12.4
- **R fixest Threads:** 1
- **Date:** 2026-01-19T19:09:40.981
## Results

| Scenario | N | FEM (s) | Regress (s) | fixest (s) | Regress/FEM | fixest/FEM |
|----------|---|---------|-------------|------------|-------------|------------|
| OLS: y ~ x1 + x2 | 10M | 0.354 | 0.508 | 0.329 | 1.436 | 0.931 |
| OLS + cluster(id2) | 10M | 0.299 | 0.622 | 0.468 | 2.082 | 1.568 |
| FE: y ~ x1 + x2 \| id1 | 10M | 0.375 | 0.573 | 0.461 | 1.529 | 1.229 |
| FE + cluster: \| id1 | 10M | 0.438 | 0.656 | 0.616 | 1.498 | 1.407 |
| FE: y ~ x1 + x2 \| id1 + id2 | 10M | 1.140 | 1.458 | 0.892 | 1.279 | 0.783 |
| Worker-Firm FE | 800K | 1.358 | 1.657 | 1.712 | 1.221 | 1.261 |
| 1 FE + cluster | 10M | 0.334 | 0.477 | 0.687 | 1.431 | 2.059 |
| 2 FE + cluster | 10M | 0.815 | 1.149 | 0.992 | 1.410 | 1.217 |
| 3 FE + cluster | 10M | 0.955 | 1.322 | 1.313 | 1.384 | 1.375 |

**Legend:**
- FEM = FixedEffectModels.jl
- Regress = Regress.jl
- fixest = R fixest package
- Ratio < 1 means faster than FEM
