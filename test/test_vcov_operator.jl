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
    using CovarianceMatrices: CovarianceMatrices, momentmatrix

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
        kw = CovarianceMatrices.bandwidthweights(k, X)
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
    y = Float64.(df.Sales)
    X = hcat(ones(nrow(df)), Float64.(df.Price), Float64.(df.NDI))
    Z = hcat(ones(nrow(df)), Float64.(df.Price), Float64.(df.CPI))
    model_m = Regress.ols(X, y)
    model_iv_m = Regress.iv(Regress.TSLS(), Z, X, y)

    # A fitted model's vcov reports the estimator that produced it.
    for fitted in (model, model_iv, model_m, model_iv_m)
        V = vcov(fitted)
        @test V isa CovarianceMatrix
        @test estimator(V) isa Regress.HR1
        @test Matrix(V) ≈ vcov(Regress.HC1(), fitted)
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

@testitem "Driscoll-Kraay vcov on Regress models" tags = [:vcov, :hac, :panel] begin
    using Regress, CSV, DataFrames, LinearAlgebra, StatsBase
    using CovarianceMatrices: CovarianceMatrices, DriscollKraay, CovarianceMatrix, bandwidth

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    n = nrow(df)
    y = df.Sales

    # The upstream implementation is the reference: these methods exist only to
    # resolve dispatch, not to compute anything of their own.
    reference(k, m; kwargs...) = invoke(
        CovarianceMatrices.vcov, Tuple{DriscollKraay, RegressionModel}, k, m; kwargs...)

    dk(rows; correction = :HC0) = DriscollKraay(
        Bartlett(4); tis = df.Year[rows], iis = df.State[rows], correction)

    model = Regress.ols(df, @formula(Sales ~ Price + NDI))
    k = dk(model.esample)

    model_iv = Regress.iv(TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin)))

    model_mat = Regress.ols([ones(n) df.Price df.NDI], y)
    model_iv_mat = Regress.iv(
        TSLS(), [ones(n) df.Price df.NDI], [ones(n) df.Pimin df.NDI], y;
        n_endogenous = 1)

    for (rows, m) in ((model.esample, model), (model_iv.esample, model_iv),
        (1:n, model_mat), (1:n, model_iv_mat))
        kk = dk(rows)
        V = vcov(kk, m)
        @test V isa CovarianceMatrix
        @test estimator(V) isa DriscollKraay
        @test Matrix(V) ≈ reference(kk, m)
        @test stderror(kk, m) ≈ sqrt.(diag(reference(kk, m)))

        # The estimator's small-sample correction reaches upstream.
        kk1 = dk(rows; correction = :HC1)
        @test Matrix(vcov(kk1, m)) ≈ reference(kk1, m)
        @test !isapprox(Matrix(vcov(kk1, m)), reference(kk, m))

        # The correction is set on the estimator, not passed to `vcov`.
        @test_throws "takes no `type` keyword" vcov(kk, m; type = :HC1)

        # The bandwidth of the time kernel is reported: as given when fixed, and the
        # one upstream selects when automatic.
        @test bandwidth(vcov(kk, m)) == 4.0
        knw = DriscollKraay(
            Bartlett{NeweyWest}(); tis = df.Year[rows], iis = df.State[rows])
        @test bandwidth(vcov(knw, m)) == bandwidth(reference(knw, m))
        @test bandwidth(vcov(m + vcov(knw))) == bandwidth(reference(knw, m))
    end

    # Driscoll-Kraay scales by the number of time periods; Regress's own HAC
    # sandwich scales by the number of observations. Reusing that sandwich would
    # return variances wrong by roughly T/n.
    @test !isapprox(Matrix(vcov(k, model)), Matrix(vcov(Bartlett(4), model)); rtol = 0.1)
end

@testitem "Driscoll-Kraay named by column on formula models" tags = [:vcov, :hac, :panel] begin
    using Regress, CSV, DataFrames, LinearAlgebra, StatsBase
    using CovarianceMatrices: CovarianceMatrices, DriscollKraay, CovarianceMatrix, aVar

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    allowmissing!(df, :Price)
    df.Price[1:7] .= missing   # a sample smaller than the table
    n = nrow(df)

    named(; correction = :HC0) = DriscollKraay(
        Bartlett(4); tis = :Year, iis = :State, correction)
    given(rows; correction = :HC0) = DriscollKraay(
        Bartlett(4); tis = df.Year[rows], iis = df.State[rows], correction)

    model = Regress.ols(df, @formula(Sales ~ Price + NDI); save_cluster = [:Year, :State])
    model_iv = Regress.iv(TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin));
        save_cluster = [:Year, :State])

    for m in (model, model_iv), correction in (:HC0, :HC1, :sss)

        k, kv = named(; correction), given(m.esample; correction)
        @test Matrix(vcov(k, m)) ≈ Matrix(vcov(kv, m))
        @test stderror(k, m) ≈ stderror(kv, m)
        @test aVar(k, m) ≈ aVar(kv, m)
        mk = m + vcov(k)
        @test stderror(mk) ≈ stderror(kv, m)
        @test Matrix(vcov(mk)) ≈ Matrix(vcov(m + vcov(kv)))
    end
    fs_named = first_stage(model_iv + vcov(named()))
    fs_given = first_stage(model_iv + vcov(given(model_iv.esample)))
    @test fs_named.F_robust ≈ fs_given.F_robust

    # Columns not saved with the model are named in the error.
    unsaved = Regress.ols(df, @formula(Sales ~ Price + NDI); save_cluster = :State)
    @test_throws "Cluster variable :Year not found" vcov(named(), unsaved)
    @test_throws "Cluster variable :Year not found" vcov(
        named(), Regress.iv(TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin))))

    # A model fitted from matrices holds no columns to look up.
    model_mat = Regress.ols([ones(n) df.NDI], df.Sales)
    @test_throws "names its time and unit indices by column, but a model fitted from matrices" vcov(
        named(), model_mat)
end

@testitem "residual adjustment delegates to CovarianceMatrices" tags = [:ols, :iv, :vcov] begin
    using Regress, CSV, DataFrames, LinearAlgebra, StatsBase
    using Regress: fe
    import CovarianceMatrices as CM

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    model = Regress.ols(df, @formula(Sales ~ Price + NDI))
    model_iv = Regress.iv(Regress.TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin)))

    # HC0 and HC2-HC5 are upstream's; only the values are pinned here, since the
    # forwarding itself is what makes them reachable.
    for k in (CM.HC0(), CM.HC2(), CM.HC3(), CM.HC4(), CM.HC5())
        for m in (model, model_iv)
            @test Regress.residualadjustment(k, m) ≈ CM.residual_adjustment(k, m)
        end
    end

    # HR1 deliberately does not delegate: upstream divides by `n - length(coef(m))`,
    # which cannot see absorbed fixed effects. Without fe(...) the two agree; with
    # fe(...) Regress's `dof_residual` is the smaller, more conservative denominator.
    @test Regress.residualadjustment(CM.HC1(), model) ≈
          CM.residual_adjustment(CM.HC1(), model)

    model_fe = Regress.ols(df, @formula(Sales ~ Price + fe(State)))
    @test dof_residual(model_fe) < nobs(model_fe) - length(coef(model_fe))
    @test Regress.residualadjustment(CM.HC1(), model_fe) >
          CM.residual_adjustment(CM.HC1(), model_fe)
end

@testitem "CR2/CR3 match the direct formula" tags = [:ols, :iv, :vcov, :cluster] begin
    using Regress, CSV, DataFrames, LinearAlgebra, StatsBase
    import CovarianceMatrices as CM

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    df.w = abs.(sin.(1:nrow(df))) .+ 0.5

    # Cluster meat in the untransformed design `Z` with residuals `e` and weights `w`:
    # each cluster contributes `Z_g' (I - H_gg)^p (w_g .* e_g)`, with p = -1/2 (CR2) or
    # p = -1 (CR3) and H_gg = Z_g inv(Z'WZ) Z_g' W_g, as in R's `sandwich::vcovCL`.
    function meat(Z, e, w, groups, p)
        B = inv(Z' * (w .* Z))
        S = zeros(size(Z, 2), size(Z, 2))
        for g in unique(groups)
            ind = findall(==(g), groups)
            Zg = Z[ind, :]
            Hgg = Zg * B * Zg' * Diagonal(w[ind])
            s = Zg' * real((I - Hgg)^p * (w[ind] .* e[ind]))
            S .+= s * s'
        end
        return S ./ size(Z, 1)
    end
    # Two-way clustering: inclusion-exclusion over the dimensions and their
    # intersection, each term with its own leverage adjustment.
    meat(Z, e, w, g1, g2, p) = meat(Z, e, w, g1, p) .+ meat(Z, e, w, g2, p) .-
                               meat(Z, e, w, tuple.(g1, g2), p)

    for (K, p) in ((CM.CR2, -1 / 2), (CM.CR3, -1))
        # A weighted model's `modelmatrix` is `√w .* X`, matched by `momentresiduals`.
        # Year clusters are not contiguous in the data; State clusters are.
        for weighted in (false, true)
            kw = weighted ? (; weights = :w) : (;)
            model = Regress.ols(df, @formula(Sales ~ Price + NDI); kw...)
            w = weighted ? df.w[model.esample] : ones(nobs(model))
            Z = modelmatrix(model) ./ sqrt.(w)
            e = CM.momentresiduals(model) ./ sqrt.(w)
            @test e ≈ residuals(model)
            for v in (:State, :Year)
                g = df[model.esample, v]
                @test Matrix(aVar(K(g), model)) ≈ meat(Z, e, w, g, p)
            end
            gs, gy = df.State[model.esample], df.Year[model.esample]
            @test Matrix(aVar(K(gs, gy), model)) ≈ meat(Z, e, w, gs, gy, p)

            # IV: the leverage blocks come from the second-stage regressors.
            model_iv = Regress.iv(Regress.TSLS(), df,
                @formula(Sales ~ NDI + (Price ~ Pimin)); kw...)
            pe = model_iv.postestimation
            Z = pe.X_fitted ./ sqrt.(w)
            e = Regress.residuals_for_vcov(model_iv) ./ sqrt.(w)
            @test CM.bread(model_iv) ≈ inv(Z' * (w .* Z))
            for v in (:State, :Year)
                g = df[model_iv.esample, v]
                @test Matrix(aVar(K(g), model_iv)) ≈ meat(Z, e, w, g, p)
            end
            @test Matrix(aVar(K(gs, gy), model_iv)) ≈ meat(Z, e, w, gs, gy, p)
        end
    end
end

@testitem "CR2/CR3 on weighted and IV models" tags = [:ols, :iv, :vcov, :cluster] begin
    using Regress, CSV, DataFrames, LinearAlgebra, StatsBase
    import CovarianceMatrices as CM

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))

    # Each successive estimator applies a larger leverage correction, so the standard
    # errors increase monotonically, with and without weights.
    for model in (Regress.ols(df, @formula(Sales ~ Price + NDI)),
        Regress.ols(df, @formula(Sales ~ Price + NDI); weights = :Pop))
        cl = df.State[model.esample]
        se = [stderror(K(cl), model)[2] for K in (CM.CR0, CM.CR1, CM.CR2, CM.CR3)]
        @test all(isfinite, se)
        @test issorted(se)
    end

    # CR2/CR3 are reachable on IV models; the leverage blocks are built from the
    # second-stage regressor matrix rather than `modelmatrix`.
    model_iv = Regress.iv(Regress.TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin)))
    cl_iv = df.State[model_iv.esample]
    se_iv = [stderror(K(cl_iv), model_iv)[2] for K in (CM.CR0, CM.CR1, CM.CR2, CM.CR3)]
    @test all(isfinite, se_iv)
    @test issorted(se_iv)
end

@testitem "IV post-estimation vcov with weights" tags = [:iv, :vcov] begin
    using Regress, CSV, DataFrames, LinearAlgebra, StatsBase
    import CovarianceMatrices as CM

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    f = @formula(Sales ~ NDI + (Price ~ Pimin))

    # Reference: R fixest, `feols(Sales ~ NDI | Price ~ Pimin, df, weights = ~Pop)`,
    # lower triangles in the order (Intercept), NDI, Price.
    function sym3(v)
        V = zeros(3, 3)
        k = 0
        for j in 1:3, i in j:3

            k += 1
            V[i, j] = V[j, i] = v[k]
        end
        return V
    end
    FX_HETERO = sym3([0.9161968845466576, -6.453024471225661e-05, -0.0043877799998184698,
        1.2123397812074779e-07, -1.2428407657284584e-05, 0.001457942615388505])
    FX_IID = sym3([0.86396333477438114, -2.846163873676756e-05, -0.0054876724459037565,
        1.1355868786062744e-07, -1.2477219187461635e-05, 0.0014881227629465264])
    FX_CL = sym3([6.8435270511043669, -0.001483835776646838, 0.095510291155292124,
        8.2045437058479111e-07, -6.8212208155359855e-05, 0.0063607397121952427])

    m = Regress.iv(Regress.TSLS(), df, f; weights = :Pop, save_cluster = :State)
    @test Matrix(vcov(m)) ≈ FX_HETERO rtol = 1e-8
    @test Matrix(vcov(CM.HC1(), m)) ≈ FX_HETERO rtol = 1e-8
    @test Matrix(vcov(m + vcov(CM.HC1()))) ≈ FX_HETERO rtol = 1e-8
    @test Matrix(vcov(CM.Uncorrelated(), m)) ≈ FX_IID rtol = 1e-8
    @test Matrix(vcov(CM.CR1(df.State), m)) ≈ FX_CL rtol = 1e-8
    @test Matrix(vcov(m + vcov(CM.CR1(:State)))) ≈ Matrix(vcov(CM.CR1(df.State), m))

    # Constant weights leave every variance unchanged.
    df.c = fill(2.5, nrow(df))
    for est in (Regress.TSLS(), Regress.LIML())
        m0 = Regress.iv(est, df, f)
        mc = Regress.iv(est, df, f; weights = :c)
        for k in (CM.HC1(), CM.HC3(), CM.CR1(df.State), CM.CR2(df.State),
            CM.Bartlett(3), CM.Bartlett{CM.Andrews}(), CM.Uncorrelated())
            @test Matrix(vcov(k, mc)) ≈ Matrix(vcov(k, m0)) rtol = 1e-10
        end
    end
end

@testitem "multiway CR2/CR3" tags = [:ols, :iv, :vcov, :cluster] begin
    using Regress, CSV, DataFrames, LinearAlgebra, StatsBase, Random
    using Regress: fe
    import CovarianceMatrices as CM

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    model = Regress.ols(df, @formula(Sales ~ Price + NDI); save_cluster = [:State, :Year])
    cl_state = df.State[model.esample]
    cl_year = df.Year[model.esample]

    # One cluster dimension reaches the elementwise-multiplier path, several reach the
    # inclusion-exclusion path. The two must agree when the dimensions coincide.
    for K in (CM.CR2, CM.CR3)
        @test stderror(K(cl_state), model) ≈ stderror(K(cl_state, cl_state), model)
    end

    # Successive estimators apply larger leverage corrections, as in the one-way case.
    se = [stderror(K(cl_state, cl_year), model)[2]
          for K in (CM.CR0, CM.CR1, CM.CR2, CM.CR3)]
    @test all(isfinite, se)
    @test issorted(se)

    # The symbol API resolves stored cluster variables to the same result.
    for K in (CM.CR2, CM.CR3)
        @test stderror(K(:State, :Year), model) ≈ stderror(K(cl_state, cl_year), model)
    end

    # Three dimensions exercise the general subset walk rather than the two-way case.
    df3 = copy(df)
    df3.Region = mod.(df3.State, 5)
    model3 = Regress.ols(df3, @formula(Sales ~ Price + NDI))
    cl3 = (df3.State[model3.esample], df3.Year[model3.esample], df3.Region[model3.esample])
    model3_iv = Regress.iv(Regress.TSLS(), df3, @formula(Sales ~ NDI + (Price ~ Pimin)))
    for K in (CM.CR2, CM.CR3)
        @test all(isfinite, stderror(K(cl3...), model3))
        @test all(isfinite, stderror(K(cl3...), model3_iv))
    end

    # Fixed effects absorbed out of the design do not disturb the leverage blocks.
    model_fe = Regress.ols(df, @formula(Sales ~ Price + NDI + fe(State)))
    cl_fe = (df.State[model_fe.esample], df.Year[model_fe.esample])
    for K in (CM.CR2, CM.CR3)
        @test all(isfinite, stderror(K(cl_fe...), model_fe))
    end

    # A cluster whose rows span the column space makes `I - H_gg` singular, which
    # throws rather than surfacing as `Inf` in the finished variance.
    rng = MersenneTwister(11)
    n = 30
    dsing = DataFrame(y = randn(rng, n), x1 = randn(rng, n), x2 = randn(rng, n),
        z = randn(rng, n))
    model_sing = Regress.ols(dsing, @formula(y ~ x1 + x2))
    singular = "needs I - H_gg to be nonsingular"
    for K in (CM.CR2, CM.CR3)
        @test_throws singular stderror(K(repeat(1:10, inner = 3), fill(1, n)), model_sing)
    end
    # The IV computation applies the same check. A regressor that is nonzero only
    # within one cluster gives that cluster a hat-matrix eigenvalue of 1.
    dsing.d = Float64.(repeat(1:10, inner = 3) .== 1)
    model_iv_sing = Regress.iv(Regress.TSLS(), dsing, @formula(y ~ x1 + d + (x2 ~ z)))
    for K in (CM.CR2, CM.CR3)
        @test_throws singular stderror(K(repeat(1:10, inner = 3)), model_iv_sing)
    end
end

@testitem "CR2/CR3 against R sandwich" tags = [:ols, :iv, :vcov, :cluster] begin
    using Regress, CSV, DataFrames, LinearAlgebra, StatsBase
    import CovarianceMatrices as CM

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    n = nrow(df)

    # Diagonals of R `sandwich::vcovCL(fit, cluster = ..., type = "HC2"/"HC3")` (sandwich
    # 3.1.1), with `fit = lm(Sales ~ NDI + Price, df, weights = ...)` or
    # `AER::ivreg(Sales ~ NDI + Price | NDI + Pimin, data = df, weights = ...)`.
    # Order: (Intercept), NDI, Price.
    R = Dict(
        (:CR2, :ols, :w, 1) =>
            [6.9408801430461873, 1.4974013866616922e-6, 1.2051637541315835e-2],
        (:CR2, :iv, :w, 1) =>
            [7.0966308211824662, 9.0197608807030392e-7, 6.8065238824069531e-3],
        (:CR2, :ols, :u, 2) =>
            [33.515303088084245, 3.3521283872509227e-6, 4.5937081221287576e-2],
        (:CR2, :ols, :w, 2) =>
            [8.5977034199088962, 1.7937603879023921e-6, 1.5259222146829400e-2],
        (:CR3, :ols, :w, 1) =>
            [7.1571325247465580, 1.8643981443444393e-6, 1.5064837259768149e-2],
        (:CR3, :iv, :w, 1) =>
            [7.5614853792483032, 1.0193914989965819e-6, 7.4854994003590294e-3],
        (:CR3, :ols, :u, 2) =>
            [35.212602404209903, 3.7180270402146302e-6, 5.2057263959472792e-2],
        (:CR3, :ols, :w, 2) =>
            [9.2613636182139665, 2.2587430382498804e-6, 1.9174262861277130e-2],
        # Two-way IV. `vcovCL` adjusts a term whose clusters are all single rows (here
        # State × Year) with `hatvalues(ivreg)`, which is not the leverage of the
        # projected regressors used for every other term. These references are R's
        # `meatCL` for State and for Year minus the State × Year term computed with the
        # projected-regressor leverage `h = w x̂'(X̂'WX̂)⁻¹x̂`, so all three terms use the
        # same hat matrix.
        (:CR2, :iv, :u, 2) =>
            [28.288903408468084, 2.0045169821290485e-6, 2.2833984323730811e-2],
        (:CR2, :iv, :w, 2) =>
            [8.7765671799573202, 1.1607183953871947e-6, 9.0880651255923935e-3],
        (:CR3, :iv, :u, 2) =>
            [29.308190101456553, 2.1995873147067393e-6, 2.5623382915253939e-2],
        (:CR3, :iv, :w, 2) =>
            [9.6581933670386722, 1.2800987399195570e-6, 9.4025936062981930e-3])

    fits = Dict()
    for (wkey, kw) in ((:u, (;)), (:w, (; weights = :Pop)))
        fits[(:ols, wkey)] = Regress.ols(df, @formula(Sales ~ NDI + Price); kw...)
        fits[(:iv, wkey)] = Regress.iv(Regress.TSLS(), df,
            @formula(Sales ~ NDI + (Price ~ Pimin)); kw...)
    end
    for ((K, mtype, wkey, d), ref) in R
        k = getproperty(CM, K)(d == 1 ? (df.State,) : (df.State, df.Year))
        @test diag(vcov(k, fits[(mtype, wkey)])) ≈ ref rtol = 1e-8
    end

    # The matrix fit uses the same weighted hat matrix.
    X = [ones(n) df.NDI df.Price]
    ols_mw = Regress.ols(X, df.Sales; weights = df.Pop)
    for K in (CM.CR2, CM.CR3), cl in ((df.State,), (df.State, df.Year))

        @test Matrix(vcov(K(cl), ols_mw)) ≈ Matrix(vcov(K(cl), fits[(:ols, :w)])) rtol = 1e-10
    end

    # A row with zero weight leaves the estimation sample, so the fit equals one without it.
    w0 = copy(Float64.(df.Pop))
    w0[1] = 0
    ols_w0 = Regress.ols(X, df.Sales; weights = w0)
    ols_drop = Regress.ols(X[2:end, :], df.Sales[2:end]; weights = w0[2:end])
    @test nobs(ols_w0) == size(X, 1) - 1
    for K in (CM.CR2, CM.CR3)
        @test Matrix(vcov(K(df.State), ols_w0)) ≈
              Matrix(vcov(K(df.State[2:end]), ols_drop)) rtol = 1e-12
    end
end

@testitem "CachedCR on every model type" tags = [:ols, :iv, :vcov, :cluster] begin
    using Regress, CSV, DataFrames, LinearAlgebra, StatsBase
    using Regress: iv, TSLS
    import CovarianceMatrices as CM
    import GLM

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    df.NDI2 = 2 .* df.NDI
    n = nrow(df)
    X = [ones(n) df.NDI df.Price]
    Z = [ones(n) df.NDI df.Pimin]
    saved = [:State, :Year]
    ols_f = Regress.ols(df, @formula(Sales ~ NDI + Price); save_cluster = saved)
    iv_f = iv(TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin)); save_cluster = saved)
    ols_m = Regress.ols(X, df.Sales)
    iv_m = iv(TSLS(), Z, X, df.Sales)
    # With a collinear regressor the OLS moment matrix keeps its column and the IV
    # moment matrix drops it; the cache must match either.
    collinear = (
        Regress.ols(df, @formula(Sales ~ NDI + NDI2 + Price); save_cluster = saved),
        iv(TSLS(), df, @formula(Sales ~ NDI + NDI2 + (Price ~ Pimin)); save_cluster = saved),
        Regress.ols([X df.NDI2], df.Sales))

    # Collinear coefficients have NaN variances, which `≈` does not equate.
    same(a, b) = isequal(isnan.(Matrix(a)), isnan.(Matrix(b))) &&
                 filter(!isnan, Matrix(a)) ≈ filter(!isnan, Matrix(b))

    by_vector = K -> (K(df.State), K((df.State, df.Year)))
    by_name = K -> (K(:State), K(:State, :Year))
    for K in (CM.CR0, CM.CR1)
        for m in (ols_f, iv_f, collinear[1], collinear[2]),
            k in (by_vector(K)..., by_name(K)...)

            @test same(vcov(CM.CachedCR(k, m), m), vcov(k, m))
        end
        for m in (ols_m, collinear[3]), k in by_vector(K)

            @test same(vcov(CM.CachedCR(k, m), m), vcov(k, m))
        end
    end
    @test same(vcov(ols_f + vcov(CM.CachedCR(CM.CR1(:State), ols_f))), vcov(CM.CR1(:State), ols_f))
    @test same(vcov(iv_f + vcov(CM.CachedCR(CM.CR1(:State), iv_f))), vcov(CM.CR1(:State), iv_f))
    # A matrix-fitted IV model computes CR0 and CR1, cached or not.
    for k in (by_vector(CM.CR0)..., by_vector(CM.CR1)...)
        @test same(vcov(CM.CachedCR(k, iv_m), iv_m), vcov(k, iv_m))
    end

    for m in (ols_f, iv_f, ols_m, iv_m)
        # A cache that CovarianceMatrices built from a GLM holds the GLM's adjustments.
        cached_glm = CM.CachedCR(CM.CR0(df.State), GLM.lm(@formula(Sales ~ NDI + Price), df))
        @test_throws "was built by CovarianceMatrices from a model of another type" vcov(
            cached_glm, m)
        for K in (CM.CR2, CM.CR3)
            @test_throws "CachedCR on a Regress model supports CR0 and CR1" CM.CachedCR(
                K(df.State), m)
        end
    end
end

@testitem "CachedCR and matrix-fitted IV cluster estimators" tags = [
    :ols, :iv, :vcov, :cluster] begin
    using Regress, CSV, DataFrames, LinearAlgebra, StatsBase
    import CovarianceMatrices as CM

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    model = Regress.ols(df, @formula(Sales ~ Price + NDI); save_cluster = [:State, :Year])
    cl = df.State[model.esample]

    # The cache holds no leverage adjustments, so CR2/CR3 are rejected rather than
    # computed without them.
    for K in (CM.CR2, CM.CR3)
        @test_throws "CachedCR on a Regress model supports CR0 and CR1" CM.CachedCR(K(cl), model)
        @test_throws "CachedCR on a Regress model supports CR0 and CR1" CM.CachedCR(K(:State), model)
        cached = CM.CachedCR(K(cl), length(coef(model)))
        @test_throws "CachedCR on a Regress model supports CR0 and CR1" vcov(cached, model)
    end

    # An IV model fitted from matrices gives the formula fit's cluster variances,
    # exactly and over-identified.
    n = nrow(df)
    X = [ones(n) df.NDI df.Price]
    for (f, Z) in ((@formula(Sales ~ NDI + (Price ~ Pimin)), [ones(n) df.NDI df.Pimin]),
        (@formula(Sales ~ NDI + (Price ~ Pimin + Pop)), [ones(n) df.NDI df.Pimin df.Pop]))
        iv_f = Regress.iv(TSLS(), df, f)
        iv_m = Regress.iv(TSLS(), Z, X, df.Sales; n_endogenous = 1)
        for K in (CM.CR0, CM.CR1, CM.CR2, CM.CR3)
            @test Matrix(vcov(K(df.State), iv_m)) ≈ Matrix(vcov(K(df.State), iv_f)) rtol=1e-12
            @test Matrix(aVar(K(df.State), iv_m)) ≈ Matrix(aVar(K(df.State), iv_f)) rtol=1e-12
        end
        @test issorted([stderror(K(df.State), iv_m)[3]
                        for K in (CM.CR0, CM.CR1, CM.CR2, CM.CR3)])
        for K in (CM.CR0, CM.CR1, CM.CR2, CM.CR3)
            k = K((df.State, df.Year))
            @test Matrix(vcov(k, iv_m)) ≈ Matrix(vcov(k, iv_f)) rtol=1e-12
        end
    end
end

@testitem "vcov entry points on every model type" tags = [:ols, :iv, :vcov, :cluster] begin
    using Regress, CSV, DataFrames, LinearAlgebra, StatsBase
    using Regress: iv, TSLS
    import CovarianceMatrices as CM

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    n = nrow(df)
    X = [ones(n) df.NDI df.Price]
    Z = [ones(n) df.NDI df.Pimin]
    ols_f = Regress.ols(df, @formula(Sales ~ NDI + Price); save_cluster = :State)
    iv_f = iv(TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin)); save_cluster = :State)
    ols_m = Regress.ols(X, df.Sales)
    iv_m = iv(TSLS(), Z, X, df.Sales)

    # Clusters by column name or by vector, with or without a keyword at its default
    # value, through `vcov`, `stderror` and `+`: one result.
    for m in (ols_f, iv_f), K in (CM.CR0, CM.CR1, CM.CR2, CM.CR3, CM.Cluster)

        V = Matrix(vcov(K(df.State), m))
        @test Matrix(vcov(K(:State), m)) == V
        @test Matrix(vcov(K(:State), m; dofadjust = true)) == V
        @test Matrix(vcov(K(df.State), m; dofadjust = true)) == V
        @test stderror(K(:State), m; dofadjust = true) == sqrt.(diag(V))
        @test Matrix(vcov(m + vcov(K(:State)))) == V
        @test Matrix(aVar(K(:State), m)) == Matrix(aVar(K(df.State), m))
    end
    for m in (ols_m, iv_m), K in (CM.CR0, CM.CR1, CM.CR2, CM.CR3, CM.Cluster)

        @test_throws "a model fitted from matrices stores no cluster data" vcov(K(:State), m)
        @test_throws "a model fitted from matrices stores no cluster data" aVar(K(:State), m)
    end
    for m in (ols_m, iv_m), K in (CM.CR0, CM.Cluster)

        @test Matrix(vcov(K(df.State), m; dofadjust = true)) == Matrix(vcov(K(df.State), m))
    end
    for m in (ols_f, iv_f, ols_m, iv_m), k in (CM.HC1(), CM.Bartlett(3))

        @test Matrix(vcov(k, m; dofadjust = true)) == Matrix(vcov(k, m))
    end

    # Keywords outside the documented ones throw.
    for m in (ols_f, iv_f, ols_m, iv_m), f in (vcov, stderror, aVar)

        @test_throws "does not accept the keyword(s) typo" f(CM.HC1(), m; typo = 1)
        @test_throws "does not accept the keyword(s) typo" f(CM.CR1(df.State), m; typo = 1)
    end
    @test_throws "does not accept the keyword(s) scale" vcov(CM.HC1(), ols_f; scale = false)

    # `dofadjust = false` drops the factor of the long-run estimators and of the iid
    # variance, and leaves the finite-sample factors that define HC1 and CR1.
    for m in (ols_f, ols_m)
        p = dof(m)
        @test Matrix(vcov(CM.Bartlett(3), m; dofadjust = false)) ≈
              Matrix(vcov(CM.Bartlett(3), m)) .* (n - p) / n
    end
    for m in (ols_f, iv_f, ols_m, iv_m)
        @test Matrix(vcov(CM.Uncorrelated(), m; dofadjust = false)) ≈
              Matrix(vcov(CM.Uncorrelated(), m)) .* dof_residual(m) / n
        @test_throws "do not apply to Uncorrelated()" vcov(CM.Uncorrelated(), m; prewhite = true)
    end
    for m in (ols_f, iv_f, ols_m, iv_m)
        @test Matrix(vcov(CM.HC1(), m; dofadjust = false)) == Matrix(vcov(CM.HC1(), m))
    end
    @test Matrix(vcov(CM.CR1(:State), ols_f; dofadjust = false)) ==
          Matrix(vcov(CM.CR1(:State), ols_f))

    # `prewhite` does not apply to cluster estimators, on any code path.
    for m in (ols_f, iv_f)
        @test_throws "not supported by CR1" vcov(CM.CR1(:State), m; prewhite = true)
    end
end

@testitem "Uncorrelated and the smoothers on every model type" tags = [:ols, :iv, :vcov] begin
    using Regress, CSV, DataFrames, LinearAlgebra, StatsBase, GLM
    using Regress: iv, TSLS, fe
    import CovarianceMatrices as CM

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    n = nrow(df)
    X = [ones(n) df.NDI df.Price]
    Z = [ones(n) df.NDI df.Pimin]
    f = @formula(Sales ~ NDI + Price)
    ols_f = Regress.ols(df, f)
    ols_m = Regress.ols(X, df.Sales)
    iv_f = iv(TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin)))
    iv_m = iv(TSLS(), Z, X, df.Sales)
    glm_f = lm(f, df)

    # `Uncorrelated` is the iid variance σ²·B with σ² = RSS/dof_residual: R's
    # `vcov(lm(...))`, with and without weights.
    for m in (ols_f, ols_m)
        @test Matrix(vcov(CM.Uncorrelated(), m)) ≈ vcov(glm_f) rtol = 1e-10
    end
    # R divides the weighted RSS by rows minus coefficients; GLM.jl's own `vcov` uses
    # the sum of the weights instead.
    ols_w = Regress.ols(df, f; weights = :Pop)
    w = Float64.(df.Pop)
    e_w = df.Sales - X * ((X' * (w .* X)) \ (X' * (w .* df.Sales)))
    V_w = sum(w .* e_w .^ 2) / (n - 3) * inv(X' * (w .* X))
    @test Matrix(vcov(CM.Uncorrelated(), ols_w)) ≈ V_w rtol = 1e-8

    # With an absorbed fixed effect, the iid variance of the slopes equals that of the
    # regression with one dummy per level.
    ols_fe = Regress.ols(df, @formula(Sales ~ NDI + Price + fe(State)))
    glm_dummies = lm(@formula(Sales ~ NDI + Price + State), df;
        contrasts = Dict(:State => DummyCoding()))
    @test Matrix(vcov(CM.Uncorrelated(), ols_fe)) ≈ vcov(glm_dummies)[2:3, 2:3] rtol = 1e-8

    # IV: σ² · (X̂'X̂)⁻¹ with X̂ the first-stage fitted regressors.
    Xhat = Z * (Z \ X)
    e = df.Sales - X * (Xhat \ df.Sales)
    V_iv = sum(abs2, e) / (n - 3) * inv(Xhat' * Xhat)
    for m in (iv_f, iv_m)
        @test Matrix(vcov(CM.Uncorrelated(), m)) ≈ V_iv rtol = 1e-10
    end

    # Collinear columns get NaN rows and columns, as in the sandwich estimators.
    df.Price2 = 2 .* df.Price
    V_col = Matrix(vcov(CM.Uncorrelated(), Regress.ols(df, @formula(Sales ~
                                                                    NDI + Price + Price2))))
    @test all(isnan, V_col[:, 4]) && all(isnan, V_col[4, :])
    @test V_col[1:3, 1:3] ≈ vcov(glm_f) rtol = 1e-10

    # `aVar` is the meat of the iid variance without the degrees-of-freedom factor.
    for m in (ols_f, ols_m, iv_f, iv_m)
        B = CM.bread(m)
        @test n * B * aVar(CM.Uncorrelated(), m) * B ≈
              Matrix(vcov(CM.Uncorrelated(), m; dofadjust = false)) rtol = 1e-10
        @test aVar(CM.Uncorrelated(), m; scale = false) ≈ n * aVar(CM.Uncorrelated(), m)
        @test_throws "do not apply to Uncorrelated()" aVar(CM.Uncorrelated(), m; demean = true)
        @test_throws "does not accept the keyword(s) typo" vcov(CM.Uncorrelated(), m; typo = 1)
    end

    # Float32 models give a Float32 iid variance.
    ols_32 = Regress.ols(df, f; double_precision = false)
    @test eltype(vcov(CM.Uncorrelated(), ols_32)) == Float32

    # The smoothers: CovarianceMatrices' value on a GLM fit, which applies the same
    # n/(n-K) factor; the formula and matrix fits agree.
    for k in (CM.UniformSmoother(3), CM.TriangularSmoother(3))
        V = vcov(k, glm_f)
        @test Matrix(vcov(k, ols_f)) ≈ V rtol = 1e-10
        @test Matrix(vcov(k, ols_m)) ≈ V rtol = 1e-10
        @test Matrix(vcov(k, iv_m)) ≈ Matrix(vcov(k, iv_f)) rtol = 1e-10
        @test Matrix(vcov(k, iv_f; dofadjust = false)) ≈
              Matrix(vcov(k, iv_f)) .* dof_residual(iv_f) / n rtol = 1e-12
        @test stderror(iv_f + vcov(k)) ≈ stderror(k, iv_f)
    end
end

@testitem "One finite-sample factor on every model type" tags = [:vcov, :cluster] begin
    using Regress, CSV, DataFrames, LinearAlgebra, StatsBase
    using Regress: fe
    import CovarianceMatrices as CM

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    n = nrow(df)
    X = [ones(n) df.NDI df.Price]
    Z = [ones(n) df.NDI df.Pimin]
    saved = [:State, :Year]
    models = (
        ols_f = Regress.ols(df, @formula(Sales ~ NDI + Price); save_cluster = saved),
        ols_m = Regress.ols(X, df.Sales),
        iv_f = Regress.iv(TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin)); save_cluster = saved),
        iv_m = Regress.iv(TSLS(), Z, X, df.Sales))
    s = df.State
    K, G = 3, 46

    # vcov(k, m) = factor · n · B · aVar(k, m) · B, with the same factor on every type.
    raw(k, m) = n .* Matrix(CM.bread(m)) * Matrix(aVar(k, m)) * Matrix(CM.bread(m))
    long_run = n / (n - K)
    factors = [
        (CM.HC0(), 1),
        (CM.CR0(s), (n - 1) / (n - K)),
        (CM.CR1(s), G / (G - 1) * (n - 1) / (n - K)),
        (CM.CR0(s, df.Year), (n - 1) / (n - K)),
        (CM.CR1(s, df.Year), 30 / 29 * (n - 1) / (n - K)),
        (CM.Cluster(s), 1),
        (CM.Bartlett(3), long_run),
        (CM.Parzen(4), long_run),
        (CM.EWC(5), long_run),
        (CM.VARHAC(), long_run)]
    for (name, m) in pairs(models), (k, c) in factors

        @test Matrix(vcov(k, m)) ≈ c .* raw(k, m) rtol = 1e-10
    end
    for (name, m) in pairs(models), k in (CM.Bartlett(3), CM.EWC(5), CM.VARHAC())

        @test Matrix(vcov(k, m; dofadjust = false)) ≈ raw(k, m) rtol = 1e-10
    end
    # CR2/CR3: the leverage adjustment is the whole small-sample correction.
    for m in (models.ols_f, models.ols_m, models.iv_f), k in (CM.CR2(s), CM.CR3(s))

        @test Matrix(vcov(k, m)) ≈ raw(k, m) rtol = 1e-10
    end

    # The formula and matrix fits agree.
    for k in (CM.HC1(), CM.HC3(), CM.CR0(s), CM.CR1(s), CM.CR1(s, df.Year), CM.Cluster(s),
        CM.Bartlett(3), CM.EWC(5))
        @test Matrix(vcov(k, models.ols_f)) ≈ Matrix(vcov(k, models.ols_m)) rtol = 1e-10
        @test Matrix(vcov(k, models.iv_f)) ≈ Matrix(vcov(k, models.iv_m)) rtol = 1e-10
    end
    @test Matrix(vcov(CM.CR2(s), models.ols_f)) ≈ Matrix(vcov(CM.CR2(s), models.ols_m)) rtol = 1e-10

    # References: R fixest `vcov(feols(...), cluster = ...)`, R sandwich
    # `vcovCL(lm/ivreg, cluster = ~State, type = "HC2"/"HC3")` and
    # `NeweyWest(lm/ivreg, lag = 2, prewhite = FALSE, adjust = TRUE)`, as lower
    # triangles. fixest orders IV coefficients (Intercept), fit_Price, NDI.
    function sym(v)
        p = round(Int, (sqrt(8 * length(v) + 1) - 1) / 2)
        V = zeros(p, p)
        c = 0
        for j in 1:p, i in j:p

            c += 1
            V[i, j] = V[j, i] = v[c]
        end
        return V
    end
    swap(V, p) = V[p, p]
    FX_OLS = sym([33.101306102670804, 0.0027106446881726368, -0.53232473252771173,
        2.2749594185295716e-06, -0.00025086602205973163, 0.029905104571217141])
    FX_IV = swap(
        sym([28.487937894214205, -0.1151242908755795, -0.00057016234798797679,
            0.014886255940823583, -0.00013579943728150474, 1.3759832857083032e-06]),
        [1, 3, 2])
    R_OLS_CR2 = sym([33.087693216089228, 0.0027665643879009123, -0.53760737479273302,
        2.3831458948573353e-06, -0.00026207275447796076, 0.031065868152410695])
    R_OLS_CR3 = sym([33.858895156437555, 0.0028891631436766731, -0.55574212342188067,
        2.5562965093482169e-06, -0.00028033334487304989, 0.0330468917708951])
    R_IV_CR2 = sym([28.41856918221513, -0.00061590134903652741, -0.10977215717738231,
        1.4291115244054496e-06, -0.00014078139009024253, 0.015349991843334026])
    R_IV_CR3 = sym([29.021861053566585, -0.00068032925345701376, -0.10659649012338861,
        1.5204169953676826e-06, -0.00014949417848441739, 0.016213449203923092])
    R_OLS_NW = sym([6.9420352565162702, 0.00013538793359429581, -0.079648518164795207,
        6.45080916441141e-07, -6.7917032405733676e-05, 0.0078878140592045388])
    R_IV_NW = sym([6.6064658057553247, -0.00017937647777714475, -0.040716770020660306,
        5.473647329840687e-07, -5.6128326000303502e-05, 0.0064231980087076646])
    # rtol = 1e-8 separates the n - 2 and n - 3 divisors (a 7e-4 difference).
    for m in (models.ols_f, models.ols_m)
        @test Matrix(vcov(CM.CR1(s), m)) ≈ FX_OLS rtol = 1e-8
        @test Matrix(vcov(CM.Bartlett(3), m)) ≈ R_OLS_NW rtol = 1e-8
    end
    for m in (models.iv_f, models.iv_m)
        @test Matrix(vcov(CM.CR1(s), m)) ≈ FX_IV rtol = 1e-8
        @test Matrix(vcov(CM.Bartlett(3), m)) ≈ R_IV_NW rtol = 1e-8
    end
    @test Matrix(vcov(CM.CR1(:State), models.iv_f + vcov(CM.CR1(:State)))) ≈ FX_IV rtol = 1e-8
    @test Matrix(vcov(models.iv_f + vcov(CM.CR1(:State)))) ≈ FX_IV rtol = 1e-8
    @test Matrix(vcov(CM.CR2(:State), models.ols_f)) ≈ R_OLS_CR2 rtol = 1e-8
    @test Matrix(vcov(CM.CR3(:State), models.ols_f)) ≈ R_OLS_CR3 rtol = 1e-8
    @test Matrix(vcov(CM.CR2(:State), models.iv_f)) ≈ R_IV_CR2 rtol = 1e-8
    @test Matrix(vcov(CM.CR3(:State), models.iv_f)) ≈ R_IV_CR3 rtol = 1e-8

    # Absorbed fixed effects: K counts the levels of a fixed effect not nested in the
    # cluster, 1 for one nested in it, and one fewer for each fixed effect after the
    # first. References: fixest, coefficients NDI, Price (IV: fit_Price, NDI).
    FX_OLS_Y = sym([3.572874362209166e-06, -0.00045874765890869791, 0.20936451595657735])
    FX_OLS_SY = sym([6.4336808300132044e-06, 1.065526439610895e-05, 0.036031395973454255])
    FX_OLS_SY2 = sym([6.4578430344072088e-06, 5.7988471344731665e-05, 0.053033114582239665])
    FX_IV_Y = swap(
        sym([
            0.42213764246917657, -0.00018653903989976047, 2.170941818130433e-06]), [2, 1])
    FX_IV_S = swap(
        sym([
            0.035849481006982327, -0.00036941457079929068, 3.8826017168438157e-06]), [2, 1])
    ols_y = Regress.ols(df, @formula(Sales ~ NDI + Price + fe(Year)); save_cluster = saved)
    ols_sy = Regress.ols(df, @formula(Sales ~ NDI + Price + fe(State) + fe(Year));
        save_cluster = saved)
    iv_y = Regress.iv(TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin) + fe(Year)); save_cluster = saved)
    iv_s = Regress.iv(TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin) + fe(State)); save_cluster = saved)
    @test Matrix(vcov(CM.CR1(:State), ols_y)) ≈ FX_OLS_Y rtol = 1e-8
    @test Matrix(vcov(CM.CR1(:State), ols_sy)) ≈ FX_OLS_SY rtol = 1e-8
    @test Matrix(vcov(CM.CR1(:State, :Year), ols_sy)) ≈ FX_OLS_SY2 rtol = 1e-8
    @test Matrix(vcov(CM.CR1(:State), iv_y)) ≈ FX_IV_Y rtol = 1e-8
    @test Matrix(vcov(CM.CR1(:State), iv_s)) ≈ FX_IV_S rtol = 1e-8

    # A Float32 model gives a Float32 variance.
    models32 = (
        Regress.ols(df, @formula(Sales ~ NDI + Price); double_precision = false),
        Regress.ols(Float32.(X), Float32.(df.Sales)),
        Regress.iv(TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin)); double_precision = false),
        Regress.iv(TSLS(), Float32.(Z), Float32.(X), Float32.(df.Sales)))
    for m in models32,
        k in (CM.HC0(), CM.HC1(), CM.HC3(), CM.CR0(s), CM.CR1(s),
            CM.Cluster(s), CM.Bartlett(3), CM.EWC(5))

        @test eltype(vcov(k, m)) === Float32
    end
end
