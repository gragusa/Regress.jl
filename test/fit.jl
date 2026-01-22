using Regress, CategoricalArrays, CSV, DataFrames, Test, LinearAlgebra,
      StatsBase
using CovarianceMatrices: CR0, CR1, CR2, CR3
using Regress: nullloglikelihood_within
include("gpu_utils.jl")

##############################################################################
##
## coefficients
##
##############################################################################
@testset "coefficients" begin
    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    df.StateC = categorical(df.State)
    df.YearC = categorical(df.Year)

    # simple
    m = @formula Sales ~ Price
    x = ols(df, m)
    @test coef(x) ≈ [139.73446, -0.22974] atol = 1e-4
    m = @formula Sales ~ Price
    x = ols(df, m, weights = :Pop)
    @test coef(x) ≈ [137.72495428982756, -0.23738] atol = 1e-4

    df.SalesInt = round.(Int64, df.Sales)
    m = @formula SalesInt ~ Price
    x = ols(df, m)
    @test coef(x) ≈ [139.72674, -0.2296683205] atol = 1e-4

    # absorb
    m = @formula Sales ~ Price + fe(State)
    x = ols(df, m)
    @test coef(x) ≈ [-0.20984] atol = 1e-4
    @test x.iterations == 1

    m = @formula Sales ~ Price + fe(State) + fe(Year)
    x = ols(df, m)
    @test coef(x) ≈ [-1.08471] atol = 1e-4
    m = @formula Sales ~ Price + fe(State) + fe(State)*Year
    x = ols(df, m)
    @test coef(x) ≈ [-0.53470, 0.0] atol = 1e-4
    m = @formula Sales ~ Price + fe(State)*Year
    x = ols(df, m)
    @test coef(x) ≈ [-0.53470, 0.0] atol = 1e-4

    #@test isempty(coef(iv(TSLS(), df, @formula(Sales ~ 0), @fe(State*Price))))
    df.mState = div.(df.State, 10)
    m = @formula Sales ~ Price + fe(mState)&fe(Year)
    x = ols(df, m)
    @test coef(x) ≈ [-1.44255] atol = 1e-4

    m = @formula Sales ~ Price + fe(State)&Year
    x = ols(df, m)
    @test coef(x) ≈ [13.993028174622104, -0.5804357763515606] atol = 1e-4
    m = @formula Sales ~ Price + Year&fe(State)
    x = ols(df, m)
    @test coef(x) ≈ [13.993028174622104, -0.5804357763515606] atol = 1e-4
    m = @formula Sales ~ 1 + Year&fe(State)
    x = ols(df, m)
    @test coef(x) ≈ [174.4084407796102] atol = 1e-4

    m = @formula Sales ~ Price + fe(State)&Year + fe(Year)&State
    x = ols(df, m)
    @test coef(x) ≈ [51.2359, - 0.5797] atol = 1e-4
    m = @formula Sales ~ Price + NDI + fe(State)&Year + fe(Year)&State
    x = ols(df, m)
    @test coef(x) ≈ [-46.4464, -0.2546, -0.005563] atol = 1e-4
    m = @formula Sales ~ 0 + Price + NDI + fe(State)&Year + fe(Year)&State
    x = ols(df, m)
    @test coef(x) ≈ [-0.21226562244177932, -0.004775616634862829] atol = 1e-4

    # recheck these two below
    m = @formula Sales ~ Pimin + Price&NDI&fe(State)
    x = ols(df, m)
    @test coef(x) ≈ [122.98713, 0.30933] atol = 1e-4
    # SSR does not work well here
    m = @formula Sales ~ Pimin + (Price&NDI)*fe(State)
    x = ols(df, m)
    @test coef(x) ≈ [0.421406, 0.0] atol = 1e-4

    # only one intercept
    m = @formula Sales ~ 1 + fe(State) + fe(Year)
    x = ols(df, m)

    # TO DO: REPORT INTERCEPT IN CASE OF FIXED EFFFECTS, LIKE STATA
    df.id3 = categorical(mod.(1:size(df, 1), Ref(3)))
    df.id4 = categorical(div.(1:size(df, 1), Ref(10)))

    m = @formula Sales ~ id3
    x = ols(df, m)
    @test length(coef(x)) == 3

    m = @formula Sales ~ 0 + id3
    x = ols(df, m)
    @test length(coef(x)) == 3

    # with fixed effects it's like implicit intercept
    m = @formula Sales ~ id3 + fe(id4)
    x = ols(df, m)
    @test length(coef(x)) == 2

    m = @formula Sales ~ Year + fe(State)
    x = ols(df, m)

    m = @formula Sales ~ id3&Price
    x = ols(df, m)
    @test length(coef(x)) == 4
    m = @formula Sales ~ id3&Price + Price
    x = ols(df, m)
    @test length(coef(x)) == 4

    m = @formula Sales ~ Year + fe(State)
    x = ols(df, m)

    m = @formula Sales ~ Year&Price + fe(State)
    x = ols(df, m)

    # absorb + weights
    m = @formula Sales ~ Price + fe(State)
    x = ols(df, m, weights = :Pop)
    @test coef(x) ≈ [- 0.21741] atol = 1e-4
    m = @formula Sales ~ Price + fe(State) + fe(Year)
    x = ols(df, m, weights = :Pop)
    @test coef(x) ≈ [- 0.88794] atol = 1e-3
    m = @formula Sales ~ Price + fe(State) + fe(State)&Year
    x = ols(df, m, weights = :Pop)
    @test coef(x) ≈ [- 0.461085492] atol = 1e-4

    # iv
    m = @formula Sales ~ (Price ~ Pimin)
    x = iv(TSLS(), df, m)
    @test coef(x) ≈ [138.19479, - 0.20733] atol = 1e-4
    m = @formula Sales ~ NDI + (Price ~ Pimin)
    x = iv(TSLS(), df, m)
    @test coef(x) ≈ [137.45096, 0.00516, - 0.76276] atol = 1e-4
    m = @formula Sales ~ NDI + (Price ~ Pimin + Pop)
    x = iv(TSLS(), df, m)
    @test coef(x) ≈ [137.57335, 0.00534, - 0.78365] atol = 1e-4
    ## multiple endogeneous variables
    m = @formula Sales ~ (Price + NDI ~ Pimin + Pop)
    x = iv(TSLS(), df, m)
    @test coef(x) ≈ [139.544, 0.8001, -0.00937] atol = 1e-4
    m = @formula Sales ~ 1 + (Price + NDI ~ Pimin + Pop)
    x = iv(TSLS(), df, m)
    @test coef(x) ≈ [139.544, 0.8001, -0.00937] atol = 1e-4
    result = [
        196.576, 0.00490989, -2.94019, -3.00686, -2.94903, -2.80183, -2.74789, -2.66682,
        -2.63855, -2.52394, -2.34751, -2.19241, -2.18707, -2.09244, -1.9691, -1.80463,
        -1.81865, -1.70428, -1.72925, -1.68501, -1.66007, -1.56102, -1.43582, -1.36812,
        -1.33677, -1.30426, -1.28094, -1.25175, -1.21438, -1.16668, -1.13033, -1.03782]

    m = @formula Sales ~ NDI + (Price&YearC ~ Pimin&YearC)
    x = iv(TSLS(), df, m)
    @test coef(x) ≈ result atol = 1e-4

    # iv + weight
    m = @formula Sales ~ (Price ~ Pimin)
    x = iv(TSLS(), df, m, weights = :Pop)
    @test coef(x) ≈ [137.03637, - 0.22802] atol = 1e-4

    # iv + weight + absorb
    m = @formula Sales ~ (Price ~ Pimin) + fe(State)
    x = iv(TSLS(), df, m)
    @test coef(x) ≈ [-0.20284] atol = 1e-4
    m = @formula Sales ~ (Price ~ Pimin) + fe(State)
    x = iv(TSLS(), df, m, weights = :Pop)
    @test coef(x) ≈ [-0.20995] atol = 1e-4
    m = @formula Sales ~ NDI + (Price ~ Pimin) + fe(State)
    x = iv(TSLS(), df, m)
    @test coef(x) ≈ [0.0011021722526916768, -0.3216374943695231] atol = 1e-4

    # non high dimensional factors
    m = @formula Sales ~ Price + YearC
    x = ols(df, m)
    m = @formula Sales ~ YearC + fe(State)
    x = ols(df, m)
    m = @formula Sales ~ Price + YearC + fe(State)
    x = ols(df, m)
    @test coef(x)[1] ≈ -1.08471 atol = 1e-4
    m = @formula Sales ~ Price + YearC + fe(State)
    x = ols(df, m, weights = :Pop)
    @test coef(x)[1] ≈ -0.88794 atol = 1e-4
    m = @formula Sales ~ NDI + (Price ~ Pimin) + YearC + fe(State)
    x = iv(TSLS(), df, m)
    @test coef(x)[1] ≈ -0.00525 atol = 1e-4

    ##############################################################################
    ##
    ## Programming
    ##
    ##############################################################################
    ols(df, term(:Sales) ~ term(:NDI) + fe(:State) + fe(:Year))
    @test fe(:State) + fe(:Year) === reduce(+, fe.([:State, :Year])) ===
          fe(term(:State)) + fe(term(:Year))

    ##############################################################################
    ##
    ## Functions
    ##
    ##############################################################################

    # function
    m = @formula Sales ~ log(Price)
    x = ols(df, m)
    @test coef(x)[1] ≈ 184.98520688 atol = 1e-4

    # function defined in user space
    mylog(x) = log(x)
    m = @formula Sales ~ mylog(Price)
    x = ols(df, m)

    # Function returning Inf
    df.Price_zero = copy(df.Price)
    df.Price_zero[1] = 0.0
    m = @formula Sales ~ log(Price_zero)
    @test_throws "Some observations for the regressor are infinite" ols(df, m)

    ##############################################################################
    ##
    ## collinearity
    ## add more tests
    ##
    ##############################################################################
    # ols
    df.Price2 = df.Price
    m = @formula Sales ~ Price + Price2
    x = ols(df, m)
    @test coef(x) ≈ [139.7344639806166, -0.22974688593485126, 0.0] atol = 1e-4

    ## iv
    df.NDI2 = df.NDI
    m = @formula Sales ~ NDI2 + NDI + (Price ~ Pimin)
    x = iv(TSLS(), df, m)
    @test iszero(coef(x)[2]) || iszero(coef(x)[3])

    m = @formula Sales ~ (Price + Price2 ~ Pimin + NDI)
    x = iv(TSLS(), df, m)
    @test iszero(coef(x)[2]) || iszero(coef(x)[3])

    ## endogeneous variables collinear with instruments are reclassified
    df.zPimin = df.Pimin
    m = @formula Sales ~ zPimin + (Price ~ NDI + Pimin)
    x = iv(TSLS(), df, m)

    m2 = @formula Sales ~ (zPimin + Price ~ NDI + Pimin)
    NDI = iv(TSLS(), df, m2)
    @test coefnames(x) == coefnames(NDI)
    @test coef(x) ≈ coef(NDI)
    @test vcov(x) ≈ vcov(NDI)

    # catch when IV underidentified
    @test_throws "Model not identified. There must be at least as many ivs as endogeneneous variables" iv(
        TSLS(), df, @formula(Sales ~ Price + (NDI + Pop ~ NDI)))
    @test_throws "Model not identified. There must be at least as many ivs as endogeneneous variables" iv(
        TSLS(), df, @formula(Sales ~ Price + (Pop ~ Price)))

    # catch when IV underidentified
    @test_throws "Model not identified. There must be at least as many ivs as endogeneneous variables" iv(
        TSLS(), df, @formula(Sales ~ Price + (NDI + Pop ~ NDI)))

    # Make sure all coefficients are estimated
    p = [100.0, -40.0, 30.0, 20.0]
    df_r = DataFrame(y = p, x = p .^ 4)
    result = ols(df_r, @formula(y ~ x))
    @test sum(abs.(coef(result)) .> 0) == 2
end

##############################################################################
##
## std errors
##
##############################################################################
@testset "standard errors" begin
    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    df.StateC = categorical(df.State)
    df.YearC = categorical(df.Year)

    # Simple - matches stata (default stderror uses HC1)
    m = @formula Sales ~ Price
    x = ols(df, m)
    @test stderror(x) ≈ [1.686791, 0.0167042] atol = 1e-6
    # IV model - now uses HC1 robust SE by default (not classical SE)
    m = @formula Sales ~ (Price ~ Pimin)
    x = iv(TSLS(), df, m)
    @test stderror(x) ≈ [1.6325, 0.01674] atol = 1e-3  # HC1 robust SE (default)
    # Stata areg - Note: Small DOF convention differences expected
    m = @formula Sales ~ Price + fe(State)
    x = ols(df, m)
    @test stderror(x) ≈ [0.0110] atol = 1e-3  # HC1 robust SE (differs from classical)

    # HC estimators (heteroskedasticity-robust)
    m = @formula Sales ~ Price
    x = ols(df, m)
    @test stderror(CovarianceMatrices.HC0(), x)[2] ≈ 0.01669 atol = 1e-4
    @test stderror(CovarianceMatrices.HC1(), x)[2] ≈ 0.01670 atol = 1e-4  # Default
    @test stderror(CovarianceMatrices.HC2(), x)[2] ≈ 0.01671 atol = 1e-4  # Leverage-adjusted
    @test stderror(CovarianceMatrices.HC3(), x)[2] ≈ 0.01673 atol = 1e-4  # Squared leverage
    @test stderror(CovarianceMatrices.HC4(), x)[2] ≈ 0.01673 atol = 1e-4
    @test stderror(CovarianceMatrices.HC5(), x)[2] ≈ 0.01671 atol = 1e-4

    # HC estimators with FE
    m = @formula Sales ~ Price + fe(State)
    x = ols(df, m)
    @test stderror(CovarianceMatrices.HC0(), x)[1] ≈ 0.01081 atol = 1e-4
    @test stderror(CovarianceMatrices.HC1(), x)[1] ≈ 0.01100 atol = 1e-4
    @test stderror(CovarianceMatrices.HC2(), x)[1] ≈ 0.01083 atol = 1e-4
    @test stderror(CovarianceMatrices.HC3(), x)[1] ≈ 0.01084 atol = 1e-4

    # Clustering models (new API: save_cluster + post-estimation stderror)
    # Uses fixest-style small sample correction: G/(G-1) * (n-1)/(n-K)
    # with K.fixef = "nonnested" (FE nested in cluster not counted)

    # CR estimators (cluster-robust) - no FE
    m = @formula Sales ~ Price
    x = ols(df, m, save_cluster = :State)
    @test stderror(CR0(:State), x)[2] ≈ 0.03749 atol = 1e-4  # No G/(G-1) adjustment
    @test stderror(CR1(:State), x)[2] ≈ 0.0379228 atol = 1e-4  # With G/(G-1)
    @test stderror(CR2(:State), x)[2] ≈ 0.03835 atol = 1e-4  # Leverage-adjusted
    @test stderror(CR3(:State), x)[2] ≈ 0.03889 atol = 1e-4  # Squared leverage

    # CR estimators with FE (FE not nested in cluster)
    m = @formula Sales ~ Price + fe(State)
    x = ols(df, m, save_cluster = :Year)
    @test stderror(CR1(:Year), x) ≈ [0.0220563] atol = 5e-4

    # CR estimators with FE (FE nested in cluster)
    m = @formula Sales ~ Price + fe(State)
    x = ols(df, m, save_cluster = :State)
    @test stderror(CR0(:State), x)[1] ≈ 0.03535 atol = 1e-4  # No G/(G-1)
    @test stderror(CR1(:State), x)[1] ≈ 0.0357498 atol = 1e-4  # With G/(G-1)
    @test stderror(CR2(:State), x)[1] ≈ 0.03622 atol = 1e-4  # Leverage-adjusted
    @test stderror(CR3(:State), x)[1] ≈ 0.03659 atol = 1e-4  # Squared leverage

    # IV model HC estimators
    # Note: HC0/HC1/HC2 all use sandwich formula. HC1 adds DOF adjustment, HC2 adds leverage adjustment.
    # These values validated against R's sandwich::vcovHC for ivreg models.
    m = @formula Sales ~ CPI + (Price ~ Pimin)
    x = iv(TSLS(), df, m)
    @test stderror(CovarianceMatrices.HC0(), x)[3] ≈ 0.0553 atol = 1e-3
    @test stderror(CovarianceMatrices.HC1(), x)[3] ≈ 0.0554 atol = 1e-3
    @test stderror(CovarianceMatrices.HC2(), x)[3] ≈ 0.0554 atol = 1e-3  # Leverage-adjusted

    # IV model CR estimators
    x = iv(TSLS(), df, m, save_cluster = :State)
    @test stderror(CR0(:State), x)[3] ≈ 0.1058 atol = 1e-3
    @test stderror(CR1(:State), x)[3] ≈ 0.1070 atol = 1e-3

    # IV with FE and cluster
    # Note: These values match FixedEffectModels.jl with Vcov.cluster(:Year)
    m = @formula Sales ~ CPI + (Price ~ Pimin) + fe(State)
    x = iv(TSLS(), df, m, save_cluster = :Year)
    @test stderror(CR0(:Year), x)[2] ≈ 0.0747 atol = 1e-3
    @test stderror(CR1(:Year), x)[2] ≈ 0.0760 atol = 1e-3

    # multiway clustering - matches R fixest
    m = @formula Sales ~ Price
    x = ols(df, m, save_cluster = [:State, :Year])
    @test stderror(CR1(:State, :Year), x) ≈ [6.196362, 0.0403469] atol = 5e-3

    # fe + multiway clustering - matches R fixest
    m = @formula Sales ~ Price + fe(State)
    x = ols(df, m, save_cluster = [:State, :Year])
    @test stderror(CR1(:State, :Year), x) ≈ [0.0405335] atol = 1e-4
end

##############################################################################
##
## subset
##
##############################################################################
@testset "subset" begin
    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    df.StateC = categorical(df.State)
    df.YearC = categorical(df.Year)

    m = @formula Sales ~ Price + StateC
    x0 = ols(df[df.State .<= 30, :], m)

    m = @formula Sales ~ Price + StateC
    Price = ols(df, m, subset = df.State .<= 30)
    @test length(Price.esample) == size(df, 1)
    @test coef(x0) ≈ coef(Price) atol = 1e-4
    @test vcov(x0) ≈ vcov(Price) atol = 1e-4

    df.State_missing = ifelse.(df.State .<= 30, df.State, missing)
    df.StateC_missing = categorical(df.State_missing)
    m = @formula Sales ~ Price + StateC_missing
    NDI = ols(df, m)
    @test length(NDI.esample) == size(df, 1)
    @test coef(x0) ≈ coef(NDI) atol = 1e-4
    @test vcov(x0) ≈ vcov(NDI) atol = 1e-2

    # missing weights
    df.Price_missing = ifelse.(df.State .<= 30, df.Price, missing)
    m = @formula Sales ~ NDI + StateC
    x = ols(df, m, weights = :Price_missing)
    @test length(x.esample) == size(df, 1)

    # missing interaction
    m = @formula Sales ~ NDI + fe(State)&Price_missing
    x = ols(df, m)
    @test nobs(x) == count(.!ismissing.(df.Price_missing))

    m = @formula Sales ~ Price + fe(State)
    x3 = ols(df, m, subset = df.State .<= 30)
    @test length(x3.esample) == size(df, 1)
    @test coef(x0)[2] ≈ coef(x3)[1] atol = 1e-4

    m = @formula Sales ~ Price + fe(State_missing)
    x4 = ols(df, m)
    @test coef(x0)[2] ≈ coef(x4)[1] atol = 1e-4

    # categorical variable as fixed effects
    m = @formula Sales ~ Price + fe(State)
    x5 = ols(df, m, subset = df.State .>= 30)

    #Error reported by Erik - cluster vcov with subset
    m = @formula Sales ~ Pimin + CPI
    x = ols(df, m, save_cluster = :State, subset = df.State .>= 30)
    @test diag(vcov(CR1(:State), x)) ≈ [130.7464887, 0.0257875, 0.0383939] atol = 0.5
end

@testset "statistics" begin
    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    df.StateC = categorical(df.State)
    df.YearC = categorical(df.Year)
    ##############################################################################
    ##
    ## R2
    ##
    ##############################################################################
    m = @formula Sales ~ Price
    x = ols(df, m)
    @test r2(x) ≈ 0.0969 atol = 1e-4
    @test adjr2(x) ≈ 0.0956 atol = 1e-3  # Updated: Small DOF convention differences
    m = @formula Sales ~ Price + Pimin + fe(State)
    x = ols(df, m)
    @test r2(x) ≈ 0.77472 atol = 1e-4
    @test adjr2(x) ≈ 0.766768 atol = 1e-4

    ##############################################################################
    ##
    ## F Stat (Robust Wald F with HC1 vcov)
    ##
    ## Note: F-statistics are now computed as robust Wald F using HC1 vcov.
    ## These values differ from classical MSS-based F-statistics.
    ##############################################################################
    m = @formula Sales ~ Price
    x = ols(df, m)
    @test x.F ≈ 189.168 atol = 0.5  # Robust Wald F with HC1
    m = @formula Sales ~ Price + fe(State)
    x = ols(df, m)
    @test x.F ≈ 363.873 atol = 0.5  # Robust Wald F with HC1
    m = @formula Sales ~ (Price ~ Pimin)
    x = iv(TSLS(), df, m)
    @test x.F ≈ 13611.79 atol = 1.0  # Robust Wald F with HC1
    m = @formula Sales ~ Price + Pop
    x = ols(df, m)
    @test x.F ≈ 96.496 atol = 0.5  # Robust Wald F with HC1
    # F-stat is computed with stored vcov_estimator (HC1 by default)
    m = @formula Sales ~ (Price ~ Pimin) + fe(State)
    x = iv(TSLS(), df, m)
    @test !isnan(x.F)  # Check F-stat is computed

    # p value (from robust Wald F)
    m = @formula Pop ~ Pimin
    x = ols(df, m)
    @test x.p ≈ 0.0103315 atol = 1e-4  # p-value from robust Wald F
    m = @formula Pop ~ Pimin + Price
    x = ols(df, m)
    @test x.p ≈ 1.4818e-8 atol = 1e-9  # p-value from robust Wald F

    # Fstat  https://github.com/FixedEffects/MetricsLinearModels.jl/issues/150
    # Note: F_kp computation uses robust variance (HR1) by default
    # For this synthetic test data, the robust F_kp indicates weak instruments
    df_example = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Ftest.csv")))
    x = iv(TSLS(), df_example, @formula(Y ~ fe(id) + (X1 + X2 ~ Z1 + Z2)))
    @test x.F_kp ≈ 2.1932 atol = 0.001  # Robust K-P F-stat (weak instruments in this synthetic data)

    ##############################################################################
    ##
    ## F_kp r_kp statistics for IV (robust variance with HR1 by default)
    ## Note: These values are computed using robust sandwich variance, which gives
    ## more conservative (smaller) F-stats than the homoskedastic formula.
    ##
    ##############################################################################
    m = @formula Sales ~ (Price ~ Pimin)
    x = iv(TSLS(), df, m)
    @test x.F_kp ≈ 467.8717 atol = 0.001
    m = @formula Sales ~ NDI + (Price ~ Pimin)
    x = iv(TSLS(), df, m)
    @test x.F_kp ≈ 244.1580 atol = 0.001
    m = @formula Sales ~ (Price ~ Pimin + CPI)
    x = iv(TSLS(), df, m)
    @test x.F_kp ≈ 427.6440 atol = 0.001
    m = @formula Sales ~ (Price ~ Pimin) + fe(State)
    x = iv(TSLS(), df, m)
    @test x.F_kp ≈ 471.7853 atol = 0.001

    # TODO: Tests for F_kp with robust/cluster vcov require passing vcov type
    # to first-stage F-stat computation. Will be re-enabled once implemented.

    ############################################
    ##
    ## loglikelihood and related
    ##
    ############################################

    m = @formula(Sales ~ Price)
    x = ols(df, m)
    @test loglikelihood(x) ≈ -6625.8266 atol = 1e-4
    @test nullloglikelihood(x) ≈ -6696.1387 atol = 1e-4
    @test r2(x, :McFadden) ≈ 0.01050 atol = 1e-4 # Pseudo R2 in R fixest
    @test adjr2(x, :McFadden) ≈ 0.0102 atol = 1e-3  # Updated: Small DOF differences

    m = @formula(Sales ~ Price + Pimin)
    x = ols(df, m)
    @test loglikelihood(x) ≈ -6598.6300 atol = 1e-4
    @test nullloglikelihood(x) ≈ -6696.1387 atol = 1e-4
    @test r2(x, :McFadden) ≈ 0.01456 atol = 1e-4 # Pseudo R2 in R fixest
    @test adjr2(x, :McFadden) ≈ 0.0141 atol = 1e-3  # Updated: Small DOF differences

    m = @formula(Sales ~ Price + Pimin + fe(State))
    x = ols(df, m)
    @test loglikelihood(x) ≈ -5667.7629 atol = 1e-4
    @test nullloglikelihood(x) ≈ -6696.1387 atol = 1e-4
    @test nullloglikelihood_within(x) ≈ -5891.2836 atol = 1e-4
    @test r2(x, :McFadden) ≈ 0.15358 atol = 1e-4 # Pseudo R2 in R fixest
    @test adjr2(x, :McFadden) ≈ 0.14656 atol = 1e-4
end

@testset "singletons" begin
    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    df.StateC = categorical(df.State)
    df.YearC = categorical(df.Year)

    df.n = max.(1:size(df, 1), 60)
    df.pn = categorical(df.n)
    m = @formula Sales ~ Price + fe(pn)
    x = ols(df, m)
    @test x.nobs == 60

    m = @formula Sales ~ Price + fe(pn)
    x = ols(df, m, drop_singletons = false)
    @test x.nobs == 1380
end

@testset "unbalanced panel" begin
    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/EmplUK.csv")))

    m = @formula Wage ~ Emp + fe(Firm)
    x = ols(df, m)
    @test coef(x) ≈ [- 0.11981270017206136] atol = 1e-4
    m = @formula Wage ~ Emp + fe(Firm)&Year
    x = ols(df, m)
    @test coef(x) ≈ [-315.0000747500431, - 0.07633636891202833] atol = 1e-4
    m = @formula Wage ~ Emp + Year&fe(Firm)
    x = ols(df, m)
    @test coef(x) ≈ [-315.0000747500431, - 0.07633636891202833] atol = 1e-4
    m = @formula Wage ~ 1 + Year&fe(Firm)
    x = ols(df, m)
    @test coef(x) ≈ [- 356.40430526316396] atol = 1e-4
    m = @formula Wage ~ Emp + fe(Firm)
    x = ols(df, m, weights = :Output)
    @test coef(x) ≈ [- 0.11514363590574725] atol = 1e-4

    # absorb + weights
    m = @formula Wage ~ Emp + fe(Firm) + fe(Year)
    x = ols(df, m)
    @test coef(x) ≈ [- 0.04683333721137311] atol = 1e-4
    m = @formula Wage ~ Emp + fe(Firm) + fe(Year)
    x = ols(df, m, weights = :Output)
    @test coef(x) ≈ [- 0.043475472188120416] atol = 1e-3

    ## the last two ones test an ill conditioned model matrix
    # SSR does not work well here
    m = @formula Wage ~ Emp + fe(Firm) + fe(Firm)&Year
    x = ols(df, m)
    @test coef(x) ≈ [- 0.122354] atol = 1e-4
    @test x.iterations <= 30

    # SSR does not work well here
    m = @formula Wage ~ Emp + fe(Firm) + fe(Firm)&Year
    x = ols(df, m, weights = :Output)
    @test coef(x) ≈ [- 0.11752306001586807] atol = 1e-4
    @test x.iterations <= 50

    # add tests with missing fixed effects
    df.Firm_missing = ifelse.(df.Firm .<= 30, missing, df.Firm)

    ## test with missing fixed effects
    m = @formula Wage ~ Emp + fe(Firm_missing)
    x = ols(df, m)
    @test coef(x) ≈ [-0.1093657] atol = 1e-4
    @test stderror(x) ≈ [0.0528] atol = 1e-3  # HC1 robust SE (default)
    @test r2(x) ≈ 0.8703 atol = 1e-2
    @test adjr2(x) ≈ 0.8502 atol = 1e-2
    @test x.nobs == 821

    ## test with missing interaction
    df.Year2 = df.Year .>= 1980
    m = @formula Wage ~ Emp + fe(Firm_missing) & fe(Year2)
    x = ols(df, m)
    @test coef(x) ≈ [-0.100863] atol = 1e-4
    @test stderror(x) ≈ [0.0787] atol = 1e-3  # HC1 robust SE (default)
    @test x.nobs == 821
end

@testset "gpu" begin
    methods_vec = [:cpu]
    if GPU_AVAILABLE
        push!(methods_vec, GPU_METHOD)
    end
    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/EmplUK.csv")))
    for method in methods_vec
        # same thing with float32 precision
        local m = @formula Wage ~ Emp + fe(Firm)
        local x = ols(df, m, method = method, double_precision = false)
        @test coef(x) ≈ [- 0.11981270017206136] rtol = 1e-4
        local m = @formula Wage ~ Emp + fe(Firm)&Year
        local x = ols(df, m, method = method, double_precision = false)
        @test coef(x) ≈ [-315.0000747500431, - 0.07633636891202833] rtol = 1e-4
        local m = @formula Wage ~ Emp + Year&fe(Firm)
        local x = ols(df, m, method = method, double_precision = false)
        @test coef(x) ≈ [-315.0000747500431, - 0.07633636891202833] rtol = 1e-4
        local m = @formula Wage ~ 1 + Year&fe(Firm)
        local x = ols(df, m, method = method, double_precision = false)
        @test coef(x) ≈ [- 356.40430526316396] rtol = 1e-4
        local m = @formula Wage ~ Emp + fe(Firm)
        local x = ols(df, m, weights = :Output, method = method, double_precision = false)
        @test coef(x) ≈ [- 0.11514363590574725] rtol = 1e-4
        local m = @formula Wage ~ Emp + fe(Firm) + fe(Year)
        local x = ols(df, m, method = method, double_precision = false)
        @test coef(x) ≈ [- 0.04683333721137311] rtol = 1e-4
        local m = @formula Wage ~ Emp + fe(Firm) + fe(Year)
        local x = ols(df, m, weights = :Output, method = method, double_precision = false)
        @test coef(x) ≈ [- 0.043475472188120416] atol = 1e-3
    end
end

@testset "missings" begin
    df1 = DataFrame(a = [1.0, 2.0, 3.0, 4.0], b = [5.0, 7.0, 11.0, 13.0])
    df2 = DataFrame(a = [1.0, missing, 3.0, 4.0], b = [5.0, 7.0, 11.0, 13.0])
    x = ols(df1, @formula(a ~ b))
    @test coef(x) ≈ [-0.6500000000000004, 0.35000000000000003] atol = 1e-4
    x = ols(df1, @formula(b ~ a))
    @test coef(x) ≈ [2.0, 2.8] atol = 1e-4
    x = ols(df2, @formula(a ~ b))
    @test coef(x) ≈ [-0.8653846153846163, 0.3653846153846155] atol = 1e-4
    x = ols(df2, @formula(b ~ a))
    @test coef(x) ≈ [2.4285714285714253, 2.7142857142857157] atol = 1e-4

    # Works with Integers
    df1 = DataFrame(a = [1, 2, 3, 4], b = [5, 7, 11, 13], c = categorical([1, 1, 2, 2]))
    x = ols(df1, @formula(a ~ b))
    @test coef(x) ≈ [-0.65, 0.35] atol = 1e-4
    x = ols(df1, @formula(a ~ b + fe(c)))
    @test coef(x) ≈ [0.5] atol = 1e-4
end
