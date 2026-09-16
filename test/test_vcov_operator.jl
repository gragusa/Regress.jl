@testitem "OLS + vcov(HC)" tags = [:ols, :vcov, :hc, :smoke] begin
    using Regress, CategoricalArrays, CSV, DataFrames, LinearAlgebra, StatsBase
    using Regress: HC0, HC1, HC2, HC3

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    df.StateC = categorical(df.State)
    df.YearC = categorical(df.Year)

    # Simple OLS
    m = @formula Sales ~ Price + NDI
    model = Regress.ols(df, m)

    for hc in [HC0(), HC1(), HC2(), HC3()]
        wrapped = model + vcov(hc)
        direct_vcov = vcov(hc, model)
        direct_se = stderror(hc, model)

        @test vcov(wrapped) ≈ direct_vcov
        @test wrapped.se ≈ direct_se
        @test wrapped.t_stats ≈ coef(model) ./ direct_se
    end
end

@testitem "OLS + vcov(HC) with FE" tags = [:ols, :vcov, :hc, :fe] begin
    using Regress, CategoricalArrays, CSV, DataFrames, LinearAlgebra, StatsBase
    using Regress: fe
    using Regress: HC0, HC1, HC2, HC3

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    df.StateC = categorical(df.State)
    df.YearC = categorical(df.Year)

    # OLS with fixed effects
    m_fe = @formula Sales ~ Price + fe(State)
    model_fe = Regress.ols(df, m_fe)

    for hc in [HC0(), HC1(), HC2(), HC3()]
        wrapped = model_fe + vcov(hc)
        direct_vcov = vcov(hc, model_fe)
        direct_se = stderror(hc, model_fe)

        @test vcov(wrapped) ≈ direct_vcov
        @test wrapped.se ≈ direct_se
    end
end

@testitem "OLS + vcov(CR)" tags = [:ols, :vcov, :cluster] begin
    using Regress, CategoricalArrays, CSV, DataFrames, LinearAlgebra, StatsBase
    using Regress: CR0, CR1, CR2, CR3

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    df.StateC = categorical(df.State)
    df.YearC = categorical(df.Year)

    # Simple OLS
    m = @formula Sales ~ Price + NDI

    # OLS with cluster
    model_cluster = Regress.ols(df, m, save_cluster = :State)

    for cr in [CR0(:State), CR1(:State), CR2(:State), CR3(:State)]
        wrapped = model_cluster + vcov(cr)
        direct_vcov = vcov(cr, model_cluster)
        direct_se = stderror(cr, model_cluster)

        @test vcov(wrapped) ≈ direct_vcov
        @test wrapped.se ≈ direct_se
    end

    # OLS with two-way clustering
    model_cluster2 = Regress.ols(df, m, save_cluster = [:State, :Year])

    wrapped = model_cluster2 + vcov(CR1(:State, :Year))
    direct_vcov = vcov(CR1(:State, :Year), model_cluster2)
    direct_se = stderror(CR1(:State, :Year), model_cluster2)

    @test vcov(wrapped) ≈ direct_vcov
    @test wrapped.se ≈ direct_se
end

@testitem "OLS + vcov(HAC)" tags = [:ols, :vcov] begin
    using Regress, CategoricalArrays, CSV, DataFrames, LinearAlgebra, StatsBase
    using Regress: Bartlett

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))

    # Simple OLS
    m = @formula Sales ~ Price + NDI
    model = Regress.ols(df, m)

    # HAC estimators
    wrapped = model + vcov(Bartlett(4))
    direct_vcov = vcov(Bartlett(4), model)
    direct_se = stderror(Bartlett(4), model)

    @test vcov(wrapped) ≈ direct_vcov
    @test wrapped.se ≈ direct_se
end

@testitem "OLS + vcov(HAC) with data-driven bandwidth" tags = [:ols, :vcov] begin
    using Regress, CSV, DataFrames, LinearAlgebra, StatsBase
    using Regress: Bartlett, Andrews, NeweyWest, aVar, bandwidth
    using CovarianceMatrices: kernelweights, momentmatrix

    # `y ~ x` on this dataset separates the weighted and unweighted bandwidths by
    # a factor of 2 (Andrews) and 4.5 (NeweyWest). On Cigar they agree to ~1e-6,
    # which would make the comparisons below pass even if the weights were dropped.
    df = CSV.read(joinpath(@__DIR__, "data", "basic_validation_df.csv"), DataFrame)
    model = Regress.ols(df, @formula(y ~ x))

    X = modelmatrix(model)
    mm = momentmatrix(model)

    for k in [Bartlett{Andrews}(), Bartlett{NeweyWest}()]
        # Bandwidth selection weights come from the model matrix, giving the
        # intercept column weight 0; an intercept-free model gets all-ones
        # weights and the distinction below disappears.
        kw = kernelweights(k, X)
        @test kw == [0.0, 1.0]

        from_model = aVar(k, model)
        with_weights = aVar(k, mm; weights = kw)
        without_weights = aVar(k, mm)

        # The model path selects the bandwidth the explicitly weighted call does.
        @test bandwidth(from_model) ≈ bandwidth(with_weights)
        @test Matrix(from_model) ≈ Matrix(with_weights)

        # ... and not the one it would select with the weights omitted.
        @test !isapprox(bandwidth(from_model), bandwidth(without_weights); rtol = 1e-3)
    end
end

@testitem "show with data-driven bandwidth" tags = [:ols, :iv, :vcov, :smoke] begin
    using Regress, CSV, DataFrames, StatsBase, Printf
    using Regress: Bartlett, Andrews, NeweyWest
    using CovarianceMatrices: bandwidth

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    model = Regress.ols(df, @formula(Sales ~ Price + NDI))
    model_iv = Regress.iv(Regress.TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin)))

    for k in [Bartlett{Andrews}(), Bartlett{NeweyWest}()]
        for m in (model, model_iv)
            wrapped = m + vcov(k)
            V = vcov(wrapped)

            # The estimator alone cannot name the bandwidth it has not selected.
            @test Regress.vcov_type_name(wrapped.vcov_estimator) == "Bartlett(auto)"

            # Paired with the estimate, it reports what was selected.
            bw = bandwidth(V)
            @test bw isa Float64
            @test bw > 0
            expected = @sprintf("Bartlett(auto: %.2f)", bw)
            @test Regress.vcov_type_name(wrapped.vcov_estimator, V) == expected

            # ... and that is what `show` prints.
            out = sprint(show, wrapped)
            @test occursin(expected, out)
        end
    end

    # An estimator that selects no bandwidth is named the same either way.
    m_hc = model + vcov(Regress.HC1())
    @test bandwidth(vcov(m_hc)) === nothing
    @test Regress.vcov_type_name(m_hc.vcov_estimator, vcov(m_hc)) == "HC1"
    @test occursin("HC1", sprint(show, m_hc))

    @test Regress.vcov_type_name(Bartlett(4)) == "Bartlett(4)"
    # A fixed bandwidth is carried by the estimator itself.
    m_fixed = model + vcov(Bartlett(4))
    @test Regress.vcov_type_name(m_fixed.vcov_estimator, vcov(m_fixed)) == "Bartlett(4)"

    # `save = :minimal` stores a placeholder variance matrix that carries no
    # estimator metadata; naming the estimator must not consult it.
    m_min = Regress.ols(df, @formula(Sales ~ Price + NDI); save = :minimal)
    @test Regress.vcov_type_name(m_min.vcov_estimator, vcov(m_min)) == "HC1"
    @test !isempty(sprint(show, m_min))
end

@testitem "IV + vcov(HC)" tags = [:iv, :vcov, :hc] begin
    using Regress, CategoricalArrays, CSV, DataFrames, LinearAlgebra, StatsBase
    using Regress: HC0, HC1, HC2

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    df.StateC = categorical(df.State)
    df.YearC = categorical(df.Year)

    # Simple IV
    m = @formula Sales ~ NDI + (Price ~ Pimin)
    model = Regress.iv(Regress.TSLS(), df, m)

    for hc in [HC0(), HC1(), HC2()]
        wrapped = model + vcov(hc)
        direct_vcov = vcov(hc, model)
        direct_se = stderror(hc, model)

        @test vcov(wrapped) ≈ direct_vcov
        @test wrapped.se ≈ direct_se
        @test wrapped.t_stats ≈ coef(model) ./ direct_se
    end
end

@testitem "IV + vcov(CR)" tags = [:iv, :vcov, :cluster] begin
    using Regress, CategoricalArrays, CSV, DataFrames, LinearAlgebra, StatsBase
    using Regress: CR0, CR1

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    df.StateC = categorical(df.State)
    df.YearC = categorical(df.Year)

    # Simple IV
    m = @formula Sales ~ NDI + (Price ~ Pimin)

    # IV with cluster
    model_cluster = Regress.iv(Regress.TSLS(), df, m, save_cluster = :State)

    for cr in [CR0(:State), CR1(:State)]
        wrapped = model_cluster + vcov(cr)
        direct_vcov = vcov(cr, model_cluster)
        direct_se = stderror(cr, model_cluster)

        @test vcov(wrapped) ≈ direct_vcov
        @test wrapped.se ≈ direct_se
    end
end

@testitem "IV + vcov(HC) with FE" tags = [:iv, :vcov, :hc, :fe] begin
    using Regress, CategoricalArrays, CSV, DataFrames, LinearAlgebra, StatsBase
    using Regress: fe
    using Regress: HC0, HC1

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    df.StateC = categorical(df.State)
    df.YearC = categorical(df.Year)

    # IV with FE
    m_fe = @formula Sales ~ (Price ~ Pimin) + fe(State)
    model_fe = Regress.iv(Regress.TSLS(), df, m_fe)

    for hc in [HC0(), HC1()]
        wrapped = model_fe + vcov(hc)
        direct_vcov = vcov(hc, model_fe)
        direct_se = stderror(hc, model_fe)

        @test vcov(wrapped) ≈ direct_vcov
        @test wrapped.se ≈ direct_se
    end
end

@testitem "IV + vcov(HAC)" tags = [:iv, :vcov] begin
    using Regress, CategoricalArrays, CSV, DataFrames, LinearAlgebra, StatsBase
    using Regress: Bartlett

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))

    # Simple IV
    m = @formula Sales ~ NDI + (Price ~ Pimin)
    model = Regress.iv(Regress.TSLS(), df, m)

    # HAC estimators for IV
    wrapped = model + vcov(Bartlett(4))
    direct_vcov = vcov(Bartlett(4), model)
    direct_se = stderror(Bartlett(4), model)

    @test vcov(wrapped) ≈ direct_vcov
    @test wrapped.se ≈ direct_se
end

@testitem "vcov chaining" tags = [:vcov, :smoke] begin
    using Regress, CategoricalArrays, CSV, DataFrames, LinearAlgebra, StatsBase
    using Regress: HC1, HC3

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))

    m = @formula Sales ~ Price + NDI
    model = Regress.ols(df, m)

    # Test chaining: (model + vcov(A)) + vcov(B) should equal model + vcov(B)
    model_hc1 = model + vcov(HC1())
    model_hc3_chained = model_hc1 + vcov(HC3())
    model_hc3_direct = model + vcov(HC3())

    @test vcov(model_hc3_chained) ≈ vcov(model_hc3_direct)
    @test model_hc3_chained.se ≈ model_hc3_direct.se
    @test model_hc3_chained.F ≈ model_hc3_direct.F
end

@testitem "StatsAPI methods" tags = [:vcov, :smoke] begin
    using Regress, CategoricalArrays, CSV, DataFrames, LinearAlgebra, StatsBase
    using Regress: HC3

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))

    m = @formula Sales ~ Price + NDI
    model = Regress.ols(df, m)
    wrapped = model + vcov(HC3())

    # Test that delegated methods return same values as original model
    @test coef(wrapped) == coef(model)
    @test coefnames(wrapped) == coefnames(model)
    @test nobs(wrapped) == nobs(model)
    @test dof(wrapped) == dof(model)
    @test dof_residual(wrapped) == dof_residual(model)
    @test r2(wrapped) == r2(model)
    @test adjr2(wrapped) == adjr2(model)
    @test residuals(wrapped) == residuals(model)
    @test fitted(wrapped) == fitted(model)

    # Test that vcov-dependent methods use updated values
    @test stderror(wrapped) == wrapped.se

    # Test confint
    ci = confint(wrapped)
    @test size(ci) == (length(coef(wrapped)), 2)
    @test all(ci[:, 1] .< coef(wrapped))
    @test all(ci[:, 2] .> coef(wrapped))

    # Test coeftable
    ct = coeftable(wrapped)
    @test length(ct.rownms) == length(coef(wrapped))
end

@testitem "first_stage with vcov" tags = [:iv, :vcov] begin
    using Regress, CategoricalArrays, CSV, DataFrames
    using Regress: HC3

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))

    m = @formula Sales ~ NDI + (Price ~ Pimin + CPI)
    model = Regress.iv(Regress.TSLS(), df, m)

    # Test first_stage from IVEstimator (default HC1)
    fs = Regress.first_stage(model)
    @test fs.vcov_type == "HR1"  # HC1 is alias for HR1
    @test fs.F_nonrobust == model.F_first_stage_nonrobust
    @test fs.F_robust == model.F_first_stage_robust
    @test fs.n_endogenous == 1
    @test fs.n_instruments == 2

    # Test first_stage from IVEstimator with updated vcov
    model_hc3 = model + vcov(HC3())
    fs_hc3 = Regress.first_stage(model_hc3)
    @test fs_hc3.vcov_type == "HR3"  # HC3 is alias for HR3
    @test fs_hc3.F_nonrobust == model_hc3.F_first_stage_nonrobust
    @test fs_hc3.F_robust == model_hc3.F_first_stage_robust

    # Robust F-stats should differ between HC1 and HC3, but non-robust should not
    @test fs.F_nonrobust == fs_hc3.F_nonrobust
    @test fs.F_robust != fs_hc3.F_robust
end

@testitem "CovarianceMatrices model interface" tags = [:vcov, :smoke] begin
    using Regress, CSV, DataFrames, LinearAlgebra, StatsAPI
    using Regress: HC0, HC1, HC2, HC3
    using CovarianceMatrices: CovarianceMatrices as CM, CovarianceMatrix

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))

    ols_model = Regress.ols(df, @formula(Sales ~ Price + NDI))
    iv_model = Regress.iv(
        Regress.TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin + CPI)))

    # Regress's own vcov methods are narrower than the RegressionModel fallback, so
    # reaching the CovarianceMatrices implementation requires invoking it directly.
    upstream_sig = Tuple{CM.AbstractAsymptoticVarianceEstimator, CM.RegressionModel}
    upstream_vcov(k, m) = invoke(CM.vcov, upstream_sig, k, m)

    for model in (ols_model, iv_model)
        p = length(coef(model))

        # Regress extends these CovarianceMatrices functions rather than defining
        # same-named ones of its own, so the model dispatches on the interface.
        @test hasmethod(CM.bread, Tuple{typeof(model)})
        @test size(CM.bread(model)) == (p, p)
        @test CM.leverage(model) == StatsAPI.leverage(model)
        @test CM.numobs(model) == nobs(model)
        @test length(CM.mask(model)) == p

        # HC0/HC1 need only bread; HC2/HC3 additionally dispatch on CM.leverage.
        for hc in (HC0(), HC1(), HC2(), HC3())
            V = upstream_vcov(hc, model)
            @test V isa CovarianceMatrix
            @test size(V) == (p, p)
            @test all(isfinite, Matrix(V))
        end
    end

    # Reachability is what this testitem pins. The CovarianceMatrices sandwich
    # applies its own finite-sample corrections, so its results are not asserted
    # to equal those of the Regress vcov methods.
end

@testitem "vcov(model) carries estimator metadata" tags = [:vcov, :smoke] begin
    using Regress, CSV, DataFrames, LinearAlgebra, StatsBase
    using CovarianceMatrices: CovarianceMatrix, estimator, bandwidth, kernelweights
    using Regress: HC0, HC3, Bartlett, Andrews, NeweyWest

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    m = @formula Sales ~ Price + NDI
    model = Regress.ols(df, m)
    model_iv = Regress.iv(Regress.TSLS(), df, @formula(Sales ~ Price + (NDI ~ CPI)))

    # A fitted model's vcov reports the estimator that produced it.
    for fitted in (model, model_iv)
        V = vcov(fitted)
        @test V isa CovarianceMatrix
        @test estimator(V) isa Regress.HR1
        # HC estimators select nothing, so there is no bandwidth to report.
        @test bandwidth(V) === nothing
        @test kernelweights(V) === nothing
    end

    for hc in (HC0(), HC3())
        V = vcov(model + vcov(hc))
        @test V isa CovarianceMatrix
        @test estimator(V) == hc
        @test bandwidth(V) === nothing
    end

    # A HAC estimate carries the bandwidth and kernel weights it selected.
    for kernel in (Bartlett{Andrews}(), Bartlett{NeweyWest}())
        V = vcov(model + vcov(kernel))
        @test V isa CovarianceMatrix
        @test bandwidth(V) isa Float64
        @test bandwidth(V) > 0
        # The intercept column is given zero weight in bandwidth selection.
        kw = kernelweights(V)
        @test kw == [0.0, 1.0, 1.0]
        @test bandwidth(V) ≈ bandwidth(Regress.aVar(kernel, model))
    end

    # A fixed bandwidth is reported as given.
    V4 = vcov(model + vcov(Bartlett(4)))
    @test bandwidth(V4) == 4.0

    # The wrapper does not disturb the estimate itself.
    @test Matrix(vcov(model + vcov(HC3()))) ≈ vcov(HC3(), model)
end
