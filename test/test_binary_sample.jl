@testitem "glmfe removes constant-outcome fixed-effect groups" tags = [:glmfe, :probit, :fe] begin
    using Regress, DataFrames
    using Regress: fe, esample
    import GLM
    using StatsAPI: coef, fitted, residuals, response, modelmatrix, nobs,
                    nulldeviance, dof_residual

    x = repeat([-1.0, 0.0, 1.0], inner = 4)
    y = [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1]
    retained = DataFrame(x = repeat(x, 3), y = repeat(y, 3),
        group = repeat([10, 20, 30], inner = 12))
    removed = DataFrame(x = collect(range(-1.0, 1.0; length = 12)),
        y = vcat(zeros(Int, 6), ones(Int, 6)),
        group = vcat(fill(40, 6), fill(50, 5), [60]))
    df = vcat(retained, removed)
    original = copy(df)
    expected_sample = vcat(trues(36), falses(12))
    expected_reasons = vcat(fill(:none, 36), fill(:fe_all_zero, 6), fill(:fe_all_one, 6))
    retained.groupcat = string.(retained.group)

    for (link, reference_link) in ((Regress.ProbitLink(), GLM.ProbitLink()),
        (Regress.LogitLink(), GLM.LogitLink()))
        m = glmfe(@formula(y ~ x + fe(group)), df; link, tolerance = 1e-11)
        reference = GLM.glm(@formula(y ~ x + groupcat), retained,
            GLM.Binomial(), reference_link; rtol = 1e-12, atol = 1e-12)
        @test m.esample == expected_sample
        @test m.drop_reasons == expected_reasons
        @test nobs(m) == 36
        @test m.fixed_effects_dof == 3
        @test dof_residual(m) == 32
        @test response(m) == retained.y
        @test size(modelmatrix(m), 1) == length(fitted(m)) == length(residuals(m)) == 36
        @test coef(m)[1] ≈ coef(reference)[2] atol=1e-7
        @test fitted(m) ≈ fitted(reference) atol=1e-7
        @test nulldeviance(m) ≈ 72log(2)
        @test esample(m, df.y) == response(m)
        @test esample(m, view(df.x, :)) == retained.x
        @test occursin("Observations removed", sprint(show, m))
        @test_throws "must match original data length" esample(m, df.y[1:3])
        @test df == original
        if link isa Regress.ProbitLink
            legacy = Regress.probit(df, @formula(y ~ x + fe(group)); tolerance = 1e-11)
            @test legacy.esample == m.esample
            @test coef(legacy) == coef(m)
        end
    end
end

@testitem "glmfe removes separated groups recursively across fixed effects" tags = [
    :glmfe, :fe] begin
    using Regress, DataFrames, StableRNGs
    using Random: randperm
    using Regress: fe
    using StatsAPI: coef, fitted, nobs

    x = repeat([-1.0, 0.0, 1.0], inner = 4)
    y = [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1]
    core = DataFrame(x = repeat(x, 4), y = repeat(y, 4),
        g = repeat([1, 1, 2, 2], inner = 12), h = repeat([1, 2, 1, 2], inner = 12))
    chain = DataFrame(x = [-0.8, 0.3, 0.6, -0.4, 0.1], y = [0, 1, 0, 1, 0],
        g = [3, 4, 4, 5, 5], h = [3, 3, 4, 4, 1])
    df = vcat(core, chain)
    expected_sample = vcat(trues(48), falses(5))
    expected_reasons = vcat(fill(:none, 48),
        [:fe_all_zero, :fe_all_one, :fe_all_zero, :fe_all_one, :fe_all_zero])
    permutation = randperm(StableRNG(20261031), nrow(df))
    for link in (Regress.ProbitLink(), Regress.LogitLink())
        reference = glmfe(@formula(y ~ x + fe(g) + fe(h)), core;
            link, tolerance = 1e-11)
        for formula in (@formula(y ~ x + fe(g) + fe(h)), @formula(y ~ x + fe(h) + fe(g)))
            m = glmfe(formula, df; link, tolerance = 1e-11)
            @test m.esample == expected_sample
            @test m.drop_reasons == expected_reasons
            @test nobs(m) == 48
            @test m.fixed_effects_dof == 3
            @test coef(m) ≈ coef(reference) atol=1e-9
            @test fitted(m) ≈ fitted(reference) atol=1e-9
            shuffled = glmfe(formula, df[permutation, :]; link, tolerance = 1e-11)
            @test shuffled.esample == expected_sample[permutation]
            @test shuffled.drop_reasons == expected_reasons[permutation]
            @test coef(shuffled) ≈ coef(reference) atol=1e-9
        end
    end
end

@testitem "Binary sample validation and fixed-effect interactions" tags = [:glmfe, :fe] begin
    using Regress, DataFrames, FixedEffects, OffsetArrays
    using StatsAPI: nobs

    df = DataFrame(x = [-1.0, 0.0, 1.0, 0.5], y = [0, 0, 1, 1], group = [1, 1, 2, 2])
    for link in (Regress.ProbitLink(), Regress.LogitLink())
        @test_throws "No observations remain" glmfe(
            @formula(y ~ x + Regress.fe(group)), df; link)
        for value in (0, 1)
            constant = DataFrame(x = df.x, y = fill(value, 4))
            @test_throws "must contain both zeros and ones" glmfe(@formula(y ~ x), constant; link)
        end
        invalid = DataFrame(x = df.x, y = [0, 2, 1, 1], group = df.group)
        @test_throws "Bernoulli response must be zero or one" glmfe(
            @formula(y ~ x + Regress.fe(group)), invalid; link)
    end

    y = [0, 0, 1, 1]
    slope_fe = FixedEffect([1, 1, 2, 2]; interaction = [-1.0, 1.0, -1.0, 1.0])
    sample, reasons = Regress._binary_sample(y, [slope_fe])
    @test all(sample)
    @test all(==(:none), reasons)
    sample, reasons = Regress._binary_sample(view(y, :), [FixedEffect([
        "a", "a", "b", "b"])])
    @test !any(sample)
    @test reasons == [:fe_all_zero, :fe_all_zero, :fe_all_one, :fe_all_one]
    @test_throws ArgumentError Regress._binary_sample(OffsetArray(y, 0:3), FixedEffect[])

    mixed = DataFrame(x = repeat([-1.0, 0.0, 1.0], inner = 4),
        y = [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    m = glmfe(@formula(y ~ x), mixed)
    @test all(m.esample)
    @test all(==(:none), m.drop_reasons)
    @test nobs(m) == nrow(mixed)
    @test_throws ArgumentError Regress.esample(m, OffsetArray(mixed.y, 0:11))
end
