@testitem "glmfe probit and logit match GLM" tags = [:probit, :glmfe, :fe] begin
    using Regress, DataFrames, StableRNGs
    using Regress: fe
    import GLM
    using StatsAPI: coef, fitted, deviance, loglikelihood

    rng = StableRNG(29102026)
    n = 1800
    x = randn(rng, n)
    group = repeat(1:6, inner = 300)
    group_effect = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]
    eta = 0.2 .+ 0.6 .* x .+ group_effect[group]
    for (link, reference_link) in ((Regress.ProbitLink(), GLM.ProbitLink()),
        (Regress.LogitLink(), GLM.LogitLink()))
        probabilities = link isa Regress.ProbitLink ? Regress.normcdf.(eta) :
                        Regress.StatsFuns.logistic.(eta)
        y = Int.(rand(rng, n) .< probabilities)
        df = DataFrame(; y, x, group, groupcat = string.(group))
        for fixed in (false, true)
            formula = fixed ? @formula(y ~ x + fe(group)) : @formula(y ~ x)
            reference_formula = fixed ? @formula(y ~ x + groupcat) : @formula(y ~ x)
            m = glmfe(formula, df; link, max_iter = 1000, tolerance = 1e-10)
            reference = GLM.glm(reference_formula, df, GLM.Binomial(), reference_link)
            @test m.rr.link === link
            @test isconcretetype(fieldtype(typeof(m), :rr))
            @test coef(m)[fixed ? 1 : 2] ≈ coef(reference)[2] atol=1e-5
            @test all(isapprox.(fitted(m), fitted(reference); atol = 1e-5))
            @test deviance(m) ≈ deviance(reference) atol=1e-6
            @test loglikelihood(m) ≈ loglikelihood(reference) atol=1e-6
            @test m.rr.one_minus_mu ≈ 1 .- fitted(m)
            @test m.rr.resid ≈ y .- fitted(m)
            if !fixed && link isa Regress.ProbitLink
                legacy = Regress.probit(df, formula; max_iter = 1000, tolerance = 1e-10)
                @test coef(legacy) == coef(m)
                @test fitted(legacy) == fitted(m)
                @test coef(glmfe(formula, df; max_iter = 1000, tolerance = 1e-10)) ==
                      coef(m)
            end
        end
    end
end

@testitem "glmfe tail probabilities and Fisher weights" tags = [:glmfe, :probit] begin
    using Regress
    using Regress: BinaryResponse, BinaryPredictorQR, GLMWorkspace
    using Regress: deviance_bernoulli, _working_response!

    setprecision(BigFloat, 2048) do
        for link in (ProbitLink(), LogitLink()), eta in (-40.0, -10.0, 0.0, 10.0, 40.0)

            p = link isa ProbitLink ? Regress.StatsFuns.normcdf(BigFloat(eta)) :
                inv(1 + exp(-BigFloat(eta)))
            for y in (0, 1)
                expected = Float64(-2 * log(y == 1 ? p : 1 - p))
                @test deviance_bernoulli(y, eta, link) ≈ expected
            end
        end
        for (T, magnitude) in ((Float32, 12), (Float64, 28)),
            link in (ProbitLink(), LogitLink())

            pp = BinaryPredictorQR{T}(ones(T, 4, 1), zeros(T, 1), zeros(T, 1))
            rr = BinaryResponse([0, 1, 0, 1], pp, :y, link)
            rr.eta .= T.([-magnitude, -magnitude, magnitude, magnitude])
            Regress.refresh_response!(rr)
            ws = GLMWorkspace(pp.X)
            _working_response!(ws, rr)
            # Relative rounding error in Gaussian tails grows with eta^2.
            tolerance = magnitude^2 * eps(T)
            for i in eachindex(rr.y)
                eta = BigFloat(rr.eta[i])
                p = link isa ProbitLink ? Regress.StatsFuns.normcdf(eta) :
                    inv(1 + exp(-eta))
                d = link isa ProbitLink ? exp(Regress.StatsFuns.normlogpdf(eta)) :
                    p * (1 - p)
                @test ws.h[i] ≈ T(d^2 / (p * (1 - p))) rtol=tolerance
                @test ws.z[i] ≈ T(eta + (rr.y[i] - p) / d) rtol=tolerance
                @test rr.resid[i] ≈ T(rr.y[i] - p) rtol=tolerance
            end
        end
        pp = BinaryPredictorQR{Float64}(ones(2, 1), [0.0], [0.0])
        rr = BinaryResponse([0, 1], pp, :y)
        rr.eta .= [-38.5, 38.5]
        Regress.refresh_response!(rr)
        ws = GLMWorkspace(pp.X)
        _working_response!(ws, rr)
        p = Regress.StatsFuns.normcdf(BigFloat(-38.5))
        d = exp(Regress.StatsFuns.normlogpdf(BigFloat(38.5)))
        @test all(isfinite, ws.z)
        @test ws.h≈fill(Float64(d^2/(p*(1-p))), 2) atol=2nextfloat(0.0) rtol=0
        @test ws.z ≈ [-38.5 - Float64(p / d), 38.5 + Float64(p / d)]
    end

    @test deviance_bernoulli(0, 1000.0, LogitLink()) == 2000
    @test deviance_bernoulli(1, -1000.0, LogitLink()) == 2000
    pp = BinaryPredictorQR{Float64}(ones(2, 1), [40.0], [40.0])
    rr = BinaryResponse([0, 1], pp, :y)
    @test isfinite(rr.deviance)
    @test_throws "outside the representable range" _working_response!(GLMWorkspace(pp.X), rr)
end

@testitem "glmfe recovers from extreme starting coefficients" tags = [:glmfe, :probit, :fe] begin
    using Regress, DataFrames
    using Regress: fe

    df = DataFrame(x = repeat([-1.0, 0.0, 1.0], inner = 4),
        y = [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    df = repeat(df, 2)
    df.group = repeat(1:2, inner = 12)
    for (link, start, expected) in ((ProbitLink(), 28.0, Regress.link_probit(0.75)),
        (LogitLink(), 400.0, log(3)))
        for fixed in (false, true)
            formula = fixed ? @formula(y ~ x + fe(group)) : @formula(y ~ x)
            m = glmfe(formula, df; link, beta0 = fixed ? [start] : [0.0, start],
                max_iter = 100, tolerance = 1e-10)
            @test last(coef(m)) ≈ expected atol=1e-7
            @test fitted(m) ≈ repeat(repeat([0.25, 0.5, 0.75], inner = 4), 2) atol=1e-7
            @test m.deviance == m.rr.deviance == deviance(m)
        end
    end
end

@testitem "glmfe links and convergence failures" tags = [:glmfe, :probit] begin
    using Regress, DataFrames

    for p in (0.01, 0.25, 0.5, 0.9, 0.99)
        @test Regress.normcdf(Regress.link_probit(p)) ≈ p
    end
    @test Regress.link_probit(0.0) == -Inf
    @test Regress.link_probit(1.0) == Inf
    @test_throws DomainError Regress.link_probit(-0.1)
    @test_throws DomainError Regress.link_probit(1.1)

    df = DataFrame(x = repeat([-1.0, 0.0, 1.0], inner = 4),
        y = [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    formula = @formula(y ~ x)
    for link in (ProbitLink(), LogitLink())
        @test_throws "Fisher scoring did not converge" glmfe(formula, df; link, max_iter = 1)
        @test_throws "max_iter must be positive" glmfe(formula, df; link, max_iter = 0)
        for tolerance in (0.0, -1.0, Inf, NaN)
            @test_throws "tolerance must be finite and positive" glmfe(
                formula, df; link, tolerance)
        end
        @test_throws "beta0 must contain only finite values" glmfe(
            formula, df; link, beta0 = [NaN, 0.0])
    end
end

@testitem "glmfe inverse links update buffers in place" tags = [:glmfe] begin
    using Regress
    using Regress: BinaryResponse, BinaryPredictorQR, inverse_link!
    for link in (Regress.ProbitLink(), Regress.LogitLink())
        pp = BinaryPredictorQR{Float64}(ones(3, 1), [0.0], [0.0])
        rr = BinaryResponse([0, 1, 1], pp, :y, link)
        buffers = (rr.mu, rr.one_minus_mu, rr.dmu_deta)
        rr.eta .= [-40.0, 0.0, 40.0]
        @test inverse_link!(rr) === rr
        @test rr.mu === buffers[1]
        @test rr.one_minus_mu === buffers[2]
        @test rr.dmu_deta === buffers[3]
        @test rr.mu[2] == rr.one_minus_mu[2] == 0.5
        @test rr.dmu_deta[2] ≈ (link isa Regress.ProbitLink ? inv(sqrt(2pi)) : 0.25)
        @test all(isfinite, rr.mu)
        @test all(isfinite, rr.one_minus_mu)
        @test all(isfinite, rr.dmu_deta)
    end
end
