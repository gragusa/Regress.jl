@testitem "IV diagnostics: first_stage_f" tags = [:iv, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using Regress: fe

    csv_path = joinpath(@__DIR__, "data", "basic_validation_df.csv")
    df = CSV.read(csv_path, DataFrame)
    df.a = categorical(df.a)

    # m11: y ~ (x ~ z1 + z2), no FE — 1 endo, overidentified
    m11 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x ~ z1 + z2)))
    fs11 = first_stage_f(m11)
    @test fs11.F_per_endo[1] ≈ 69.71611 atol = 0.01
    @test fs11.df1 == 2
    @test fs11.df2 == 497

    # m21: y ~ (x ~ z1 + z2) + fe(a), FE — 1 endo, overidentified
    m21 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x ~ z1 + z2) + fe(a)))
    fs21 = first_stage_f(m21)
    @test fs21.F_per_endo[1] ≈ 60.65408 atol = 0.01
    @test fs21.df1 == 2
    @test fs21.df2 == 398

    # m12: y ~ (x + x2 ~ z1 + z2), no FE — 2 endo, exactly identified
    m12 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x + x2 ~ z1 + z2)))
    fs12 = first_stage_f(m12)
    @test fs12.F_per_endo[1] ≈ 69.71611 atol = 0.01  # x
    @test fs12.F_per_endo[2] ≈ 0.95367 atol = 0.01   # x2
    @test fs12.df1 == 2
    @test fs12.df2 == 497

    # m22: y ~ (x + x2 ~ z1 + z2) + fe(a), FE — 2 endo, exactly identified
    m22 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x + x2 ~ z1 + z2) + fe(a)))
    fs22 = first_stage_f(m22)
    @test fs22.F_per_endo[1] ≈ 60.65408 atol = 0.01  # x
    @test fs22.F_per_endo[2] ≈ 0.43385 atol = 0.01   # x2
    @test fs22.df1 == 2
    @test fs22.df2 == 398
end

@testitem "IV diagnostics: wu_hausman" tags = [:iv, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using Regress: fe

    csv_path = joinpath(@__DIR__, "data", "basic_validation_df.csv")
    df = CSV.read(csv_path, DataFrame)
    df.a = categorical(df.a)

    # m11: 1 endo, no FE
    m11 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x ~ z1 + z2)))
    wh11 = wu_hausman(m11)
    @test wh11.stat ≈ 15.03388 atol = 0.01
    @test wh11.p ≈ 1.198e-4 atol = 1e-5
    @test wh11.df1 == 1
    @test wh11.df2 == 497

    # m21: 1 endo, FE
    m21 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x ~ z1 + z2) + fe(a)))
    wh21 = wu_hausman(m21)
    @test wh21.stat ≈ 8.29012 atol = 0.01
    @test wh21.p ≈ 0.004201 atol = 1e-4
    @test wh21.df1 == 1
    @test wh21.df2 == 398

    # m12: 2 endo, no FE
    m12 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x + x2 ~ z1 + z2)))
    wh12 = wu_hausman(m12)
    @test wh12.stat ≈ 7.56721 atol = 0.01
    @test wh12.p ≈ 5.792e-4 atol = 1e-4
    @test wh12.df1 == 2
    @test wh12.df2 == 495

    # m22: 2 endo, FE
    m22 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x + x2 ~ z1 + z2) + fe(a)))
    wh22 = wu_hausman(m22)
    @test wh22.stat ≈ 4.37336 atol = 0.01
    @test wh22.p ≈ 0.013223 atol = 1e-3
    @test wh22.df1 == 2
    @test wh22.df2 == 396
end

@testitem "IV diagnostics: sargan" tags = [:iv, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using Regress: fe

    csv_path = joinpath(@__DIR__, "data", "basic_validation_df.csv")
    df = CSV.read(csv_path, DataFrame)
    df.a = categorical(df.a)

    # m11: 1 endo, 2 instruments — overidentified
    m11 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x ~ z1 + z2)))
    s11 = sargan(m11)
    @test s11.stat ≈ 0.19677 atol = 0.01
    @test s11.p ≈ 0.657342 atol = 0.01
    @test s11.df == 1

    # m21: 1 endo, 2 instruments, FE — overidentified
    m21 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x ~ z1 + z2) + fe(a)))
    s21 = sargan(m21)
    @test s21.stat ≈ 0.53469 atol = 0.01
    @test s21.p ≈ 0.464644 atol = 0.01
    @test s21.df == 1

    # m12: 2 endo, 2 instruments — exactly identified, should error
    m12 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x + x2 ~ z1 + z2)))
    @test_throws ErrorException sargan(m12)

    # m22: 2 endo, 2 instruments, FE — exactly identified, should error
    m22 = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x + x2 ~ z1 + z2) + fe(a)))
    @test_throws ErrorException sargan(m22)
end
