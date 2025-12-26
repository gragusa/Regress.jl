using Regress, CategoricalArrays, CSV, DataFrames, Test, LinearAlgebra, StatsBase
using CovarianceMatrices: HC0, HC1, HC2, HC3, CR0, CR1, CR2, CR3, Bartlett

##############################################################################
##
## model + vcov(estimator) pattern tests
##
## Verify that (model + vcov(estimator)).vcov_matrix ≈ vcov(estimator, model)
##
##############################################################################

@testset "OLS + vcov consistency" begin
    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    df.StateC = categorical(df.State)
    df.YearC = categorical(df.Year)

    # Simple OLS
    m = @formula Sales ~ Price + NDI
    model = ols(df, m)

    @testset "HC estimators" begin
        for hc in [HC0(), HC1(), HC2(), HC3()]
            wrapped = model + vcov(hc)
            direct_vcov = vcov(hc, model)
            direct_se = stderror(hc, model)

            @test wrapped.vcov_matrix ≈ direct_vcov
            @test wrapped.se ≈ direct_se
            @test wrapped.t_stats ≈ coef(model) ./ direct_se
        end
    end

    # OLS with fixed effects
    m_fe = @formula Sales ~ Price + fe(State)
    model_fe = ols(df, m_fe)

    @testset "HC estimators with FE" begin
        for hc in [HC0(), HC1(), HC2(), HC3()]
            wrapped = model_fe + vcov(hc)
            direct_vcov = vcov(hc, model_fe)
            direct_se = stderror(hc, model_fe)

            @test wrapped.vcov_matrix ≈ direct_vcov
            @test wrapped.se ≈ direct_se
        end
    end

    # OLS with cluster
    model_cluster = ols(df, m, save_cluster = :State)

    @testset "CR estimators" begin
        for cr in [CR0(:State), CR1(:State), CR2(:State), CR3(:State)]
            wrapped = model_cluster + vcov(cr)
            direct_vcov = vcov(cr, model_cluster)
            direct_se = stderror(cr, model_cluster)

            @test wrapped.vcov_matrix ≈ direct_vcov
            @test wrapped.se ≈ direct_se
        end
    end

    # OLS with two-way clustering
    model_cluster2 = ols(df, m, save_cluster = [:State, :Year])

    @testset "Two-way CR estimators" begin
        wrapped = model_cluster2 + vcov(CR1(:State, :Year))
        direct_vcov = vcov(CR1(:State, :Year), model_cluster2)
        direct_se = stderror(CR1(:State, :Year), model_cluster2)

        @test wrapped.vcov_matrix ≈ direct_vcov
        @test wrapped.se ≈ direct_se
    end

    # HAC estimators
    @testset "HAC estimators" begin
        wrapped = model + vcov(Bartlett(4))
        direct_vcov = vcov(Bartlett(4), model)
        direct_se = stderror(Bartlett(4), model)

        @test wrapped.vcov_matrix ≈ direct_vcov
        @test wrapped.se ≈ direct_se
    end
end

@testset "IV + vcov consistency" begin
    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    df.StateC = categorical(df.State)
    df.YearC = categorical(df.Year)

    # Simple IV
    m = @formula Sales ~ NDI + (Price ~ Pimin)
    model = iv(TSLS(), df, m)

    @testset "HC estimators" begin
        for hc in [HC0(), HC1(), HC2()]
            wrapped = model + vcov(hc)
            direct_vcov = vcov(hc, model)
            direct_se = stderror(hc, model)

            @test wrapped.vcov_matrix ≈ direct_vcov
            @test wrapped.se ≈ direct_se
            @test wrapped.t_stats ≈ coef(model) ./ direct_se
        end
    end

    # IV with cluster
    model_cluster = iv(TSLS(), df, m, save_cluster = :State)

    @testset "CR estimators for IV" begin
        for cr in [CR0(:State), CR1(:State)]
            wrapped = model_cluster + vcov(cr)
            direct_vcov = vcov(cr, model_cluster)
            direct_se = stderror(cr, model_cluster)

            @test wrapped.vcov_matrix ≈ direct_vcov
            @test wrapped.se ≈ direct_se
        end
    end

    # IV with FE
    m_fe = @formula Sales ~ (Price ~ Pimin) + fe(State)
    model_fe = iv(TSLS(), df, m_fe)

    @testset "HC estimators for IV with FE" begin
        for hc in [HC0(), HC1()]
            wrapped = model_fe + vcov(hc)
            direct_vcov = vcov(hc, model_fe)
            direct_se = stderror(hc, model_fe)

            @test wrapped.vcov_matrix ≈ direct_vcov
            @test wrapped.se ≈ direct_se
        end
    end

    # HAC estimators for IV
    @testset "HAC estimators for IV" begin
        wrapped = model + vcov(Bartlett(4))
        direct_vcov = vcov(Bartlett(4), model)
        direct_se = stderror(Bartlett(4), model)

        @test wrapped.vcov_matrix ≈ direct_vcov
        @test wrapped.se ≈ direct_se
    end
end

@testset "OLSEstimator vcov chaining" begin
    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))

    m = @formula Sales ~ Price + NDI
    model = ols(df, m)

    # Test chaining: (model + vcov(A)) + vcov(B) should equal model + vcov(B)
    model_hc1 = model + vcov(HC1())
    model_hc3_chained = model_hc1 + vcov(HC3())
    model_hc3_direct = model + vcov(HC3())

    @test model_hc3_chained.vcov_matrix ≈ model_hc3_direct.vcov_matrix
    @test model_hc3_chained.se ≈ model_hc3_direct.se
    @test model_hc3_chained.F ≈ model_hc3_direct.F
end

@testset "OLSEstimator with vcov StatsAPI methods" begin
    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))

    m = @formula Sales ~ Price + NDI
    model = ols(df, m)
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
    @test vcov(wrapped) == wrapped.vcov_matrix
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

@testset "first_stage with updated vcov" begin
    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))

    m = @formula Sales ~ NDI + (Price ~ Pimin + CPI)
    model = iv(TSLS(), df, m)

    # Test first_stage from IVEstimator (default HC1)
    fs = first_stage(model)
    @test fs.vcov_type == "HR1"  # HC1 is alias for HR1
    @test fs.F_joint == model.F_kp
    @test fs.n_endogenous == 1
    @test fs.n_instruments == 2

    # Test first_stage from IVEstimator with updated vcov
    model_hc3 = model + vcov(HC3())
    fs_hc3 = first_stage(model_hc3)
    @test fs_hc3.vcov_type == "HR3"  # HC3 is alias for HR3
    @test fs_hc3.F_joint == model_hc3.F_kp

    # F-stats should differ between HC1 and HC3
    @test fs.F_joint != fs_hc3.F_joint
end
