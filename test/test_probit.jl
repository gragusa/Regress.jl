@testitem "Probit coefficients - RWM saved results" tags = [:probit, :fe] begin
    using Regress, CSV, DataFrames
    using Regress: fe, probit
    using StatsAPI: coef

    rwm_data = CSV.read(
        joinpath(@__DIR__, "data/rwm.data"),
        DataFrame;
        header = false,
        delim = ' ',
        ignorerepeated = true
    )

    rename!(rwm_data,
        [
            :id, :female, :year, :age, :hsat, :handdum, :handper,
            :hhninc, :hhkids, :educ, :married, :haupts, :reals,
            :fachhs, :abitur, :univ, :working, :bluec, :whitec,
            :self, :beamt, :docvis, :hospvis, :public, :addon
        ])

    rwm_data[!, :visit_dummy] = ifelse.(rwm_data.docvis .> 0, 1, 0)

    model = probit(
        rwm_data,
        @formula(visit_dummy ~ age + hhninc + hhkids + educ + married + fe(id) + fe(year));
        beta0 = nothing,
        max_iter = 1000,
        tolerance = 1e-6
    )

    expected_coef = [
        0.0,
        -2.028526040781942e-6,
        -0.030016603889186432,
        -0.06974646303214804,
        -0.02956180984977514
    ]

    @test coef(model) ≈ expected_coef atol = 1e-4
    @test nobs(model) == 16312
    @test count(!, model.esample) == 11014
    @test all(==(:none), model.drop_reasons[model.esample])
    @test all(!=(:none), model.drop_reasons[.!model.esample])
end

@testitem "Probit coefficients - RWM model variants" tags = [:probit, :fe] begin
    using Regress, CSV, DataFrames
    using Regress: fe, probit
    using StatsAPI: coef

    rwm_data = CSV.read(
        joinpath(@__DIR__, "data/rwm.data"),
        DataFrame;
        header = false,
        delim = ' ',
        ignorerepeated = true
    )

    rename!(rwm_data,
        [
            :id, :female, :year, :age, :hsat, :handdum, :handper,
            :hhninc, :hhkids, :educ, :married, :haupts, :reals,
            :fachhs, :abitur, :univ, :working, :bluec, :whitec,
            :self, :beamt, :docvis, :hospvis, :public, :addon
        ])

    rwm_data[!, :visit_dummy] = ifelse.(rwm_data.docvis .> 0, 1, 0)

    model_id_fe = probit(
        rwm_data,
        @formula(visit_dummy ~ age + hhninc + hhkids + educ + married + fe(id));
        beta0 = nothing,
        max_iter = 1000,
        tolerance = 1e-6
    )
    expected_id_fe_coef = [
        0.0625323,
        -3.43276e-6,
        -0.0482711,
        -0.0721936,
        -0.0327782
    ]
    @test coef(model_id_fe) ≈ expected_id_fe_coef atol = 1e-4

    model_no_fe = probit(
        rwm_data,
        @formula(visit_dummy ~ age + hhninc + hhkids + educ + married);
        beta0 = nothing,
        max_iter = 1000,
        tolerance = 1e-6
    )
    # Without fixed effects the model carries an explicit intercept.
    expected_no_fe_coef = [
        0.1550025,
        0.0128346,
        -1.164312e-5,
        -0.1411836,
        -0.0281153,
        0.0522604
    ]
    @test coefnames(model_no_fe) ==
          ["(Intercept)", "age", "hhninc", "hhkids", "educ", "married"]
    @test coef(model_no_fe) ≈ expected_no_fe_coef atol = 1e-4

    model_educ = probit(
        rwm_data,
        @formula(visit_dummy ~ educ);
        beta0 = nothing,
        max_iter = 1000,
        tolerance = 1e-6
    )
    @test coef(model_educ) ≈ [0.8038687, -0.0417852] atol = 1e-4
end

@testitem "Probit step-halving reduces bad overshoot" tags = [:probit] begin
    using Regress
    using Regress: BinaryEstimator, BinaryPredictorQR, BinaryResponse
    using Regress: stephalving!, refresh_response!
    using StatsAPI: deviance

    X = reshape([-1.0, 1.0, 1.0], :, 1)
    y = [0.0, 1.0, 0.0]

    pp = BinaryPredictorQR{Float64}(X, [0.0], [20.0])

    rr = BinaryResponse(y, pp, :y)
    rr.deviance = deviance(rr)

    formula = @formula(y ~ x)

    m = BinaryEstimator{Float64}(
        rr,
        pp,
        formula,
        formula,
        formula,
        length(y),
        1,
        rr.deviance,
        rr.deviance,
        false,
        0,
        ["x"],
        trues(1)
    )

    alpha = zeros(length(y))
    eta_old = copy(rr.eta)

    rr.deviance_new = refresh_response!(rr, pp.X, pp.beta_new, alpha)

    @test rr.deviance_new > rr.deviance

    stephalving!(m, eta_old)

    @test rr.deviance_new <= rr.deviance
    @test pp.beta_new ≈ [0.625]
end

@testitem "Binary step-halving shrinks fixed effects and reports failure" tags = [
    :glmfe, :probit] begin
    using Regress
    using Regress: BinaryEstimator, BinaryPredictorQR, BinaryResponse, update_response!

    formula = @formula(y ~ x)
    for link in (ProbitLink(), LogitLink())
        pp = BinaryPredictorQR{Float64}(zeros(4, 1), [0.0], [0.0])
        rr = BinaryResponse([0, 1, 1, 1], pp, :y, link)
        m = BinaryEstimator(rr, pp, formula, formula, formula, 4, 1,
            rr.deviance, rr.deviance, true, 1, ["x"], trues(1))
        accepted_deviance = rr.deviance
        alpha = fill(100.0, 4)
        update_response!(m, alpha)
        @test 0 < first(rr.eta) < 100
        @test rr.deviance_new <= accepted_deviance
        @test rr.deviance == accepted_deviance
        @test pp.beta == [0.0]
        @test alpha == fill(100.0, 4)

        rr.deviance = Regress.refresh_response!(rr, pp.X, pp.beta, zeros(4))
        @test_throws "Step-halving failed" update_response!(m, alpha; max_halvings = 1)
        @test rr.deviance == accepted_deviance
    end
end

@testitem "Probit without fixed effects matches GLM" tags = [:probit] begin
    using Regress
    using Regress: probit
    using StatsAPI: coef
    using DataFrames
    using GLM  # re-exports Normal, cdf, Binomial, ProbitLink
    using StableRNGs

    rng = StableRNG(20240607)
    n = 2000
    x1 = randn(rng, n)
    x2 = randn(rng, n)
    eta = 0.4 .- 0.8 .* x1 .+ 0.5 .* x2
    y = Int.(rand(rng, n) .< GLM.Distributions.cdf.(GLM.Distributions.Normal(), eta))
    df = DataFrame(; y, x1, x2)

    m = probit(
        df, @formula(y ~ x1 + x2); beta0 = nothing, max_iter = 1000, tolerance = 1e-8)
    g = glm(@formula(y ~ x1 + x2), df, Binomial(), GLM.ProbitLink())

    # The no-fixed-effect model carries an explicit intercept.
    @test coefnames(m) == ["(Intercept)", "x1", "x2"]
    @test coef(m) ≈ coef(g) atol = 1e-5
end
