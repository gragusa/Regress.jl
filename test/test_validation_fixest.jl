# Validation tests comparing Regress.jl with R fixest package
#
# This test suite validates OLS and IV estimation with:
# - Fixed effects (none, fe1, fe1+fe2)
# - Variance estimators (HC1, cluster_fe1, two-way cluster)
# - Continuous and categorical instruments
#
# Reference values are hardcoded from R fixest 0.14.1, as written by
# test/validation/fixest_validation.R to fixest_validation_summary.csv.
# Input data: fixest_validation_data.csv (1000 observations)
#
# Tolerances for comparison (defined inline in each test):
# - RTOL_COEF = 1e-6        (Coefficients match closely)
# - RTOL_SE_HC = 1e-6       (HC standard errors)
# - RTOL_SE_CLUSTER = 1e-6  (Cluster SE - DOF/bias correction differences)
# - RTOL_R2 = 1e-6          (R-squared)

@testitem "fixest: OLS no FE" tags = [:ols, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using Regress: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: (Intercept), x1, x2, endo
    COEF = [1.122758946375919, 0.35942255387669253, 0.3065462383737957, 2.4511777942460586]
    R2 = 0.8156916809686285

    # Standard errors by vcov type
    SE_HC1 = [0.06034190702010534, 0.055780511241274776,
        0.053236727664999275, 0.03937187541081476]
    SE_CLUSTER_FE1 = [
        0.14442283634976316, 0.06589482776785886, 0.059721840781384546, 0.05265392091133568]
    SE_CLUSTER_FE1_FE2 = [
        0.39811511143940087, 0.0675199056837237, 0.05564899828003804, 0.05379002563965997]

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = Regress.ols(df, @formula(y ~ x1 + x2 + endo))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = Regress.ols(df, @formula(y ~ x1 + x2 + endo), save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = Regress.ols(df, @formula(y ~ x1 + x2 + endo), save_cluster = [:fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: OLS with fe1" tags = [:ols, :fe, :validation] begin
    using Regress
    using Regress: fe
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using Regress: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: x1, x2, endo (no intercept with FE)
    COEF = [0.41789091831165126, 0.29528228434158416, 2.301759403880162]
    R2 = 0.8400630593144001

    # Standard errors by vcov type
    SE_HC1 = [0.05431615853068957, 0.049263922969737084, 0.03876370668472496]
    SE_CLUSTER_FE1 = [0.05940211270765165, 0.05560908976997833, 0.03089648475322476]
    SE_CLUSTER_FE1_FE2 = [0.05709997116622664, 0.04378046588930851, 0.047120302730799625]

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = Regress.ols(df, @formula(y ~ x1 + x2 + endo + fe(fe1)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = Regress.ols(df, @formula(y ~ x1 + x2 + endo + fe(fe1)), save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = Regress.ols(df, @formula(y ~ x1 + x2 + endo + fe(fe1)), save_cluster = [
        :fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: OLS with fe1+fe2" tags = [:ols, :fe, :validation] begin
    using Regress
    using Regress: fe
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using Regress: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: x1, x2, endo (no intercept with FE)
    COEF = [0.3917752015810385, 0.298364514180517, 2.3318476586831953]
    R2 = 0.9316161520094017

    # Standard errors by vcov type
    SE_HC1 = [0.0350411072149658, 0.032885861336164675, 0.026075254232056692]
    SE_CLUSTER_FE1 = [0.03793439484382573, 0.04196275325188498, 0.018278597518572498]
    SE_CLUSTER_FE1_FE2 = [0.03311809661530514, 0.04098378676772609, 0.01906479240456021]

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = Regress.ols(df, @formula(y ~ x1 + x2 + endo + fe(fe1) + fe(fe2)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = Regress.ols(df, @formula(y ~ x1 + x2 + endo + fe(fe1) + fe(fe2)), save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = Regress.ols(df, @formula(y ~ x1 + x2 + endo + fe(fe1) + fe(fe2)), save_cluster = [
        :fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: IV continuous no FE" tags = [:iv, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using Regress: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6
    RTOL_F_IID = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: (Intercept), x1, x2, fit_endo
    COEF = [1.426078709850737, 0.5022110369396626, 0.31777225217776084, 2.0533383199942645]
    R2 = 0.7970527457662797
    F_IID = 290.9503015727274

    # Standard errors by vcov type
    # Note: HC1 SE computed with DOF convention that counts intercept (dof_res = n - k)
    SE_HC1 = [
        0.08893165980839529, 0.06513596685507532, 0.05498467927980058, 0.08847046479733618]
    SE_CLUSTER_FE1 = [
        0.17132281425973053, 0.08148956701313283, 0.05953477002730283, 0.07956739241725953]
    SE_CLUSTER_FE1_FE2 = [
        0.4256690844656033, 0.07470335498392455, 0.050224804551345203, 0.07447933461449091]

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = Regress.iv(Regress.TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # First-stage F-statistic (IID)
    @test Regress.first_stage(m).F_nonrobust[1] ≈ F_IID rtol = RTOL_F_IID

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = Regress.iv(Regress.TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous)), save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = Regress.iv(Regress.TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous)),
        save_cluster = [
            :fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: IV continuous fe1" tags = [:iv, :fe, :validation] begin
    using Regress
    using Regress: fe
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using Regress: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6
    RTOL_F_IID = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: x1, x2, fit_endo (no intercept with FE)
    COEF = [0.517192834221396, 0.3000057923783143, 2.0316203354076725]
    R2 = 0.832384325466159
    F_IID = 320.60480351691064

    # Standard errors by vcov type
    SE_HC1 = [0.06077896542330189, 0.04991147808908014, 0.07880872705581633]
    SE_CLUSTER_FE1 = [0.07780294115714294, 0.0565941713724285, 0.08155125433801914]
    SE_CLUSTER_FE1_FE2 = [0.0659925475162818, 0.04171831488689911, 0.06759295302443227]

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = Regress.iv(Regress.TSLS(), df, @formula(y ~
                                                x1 + x2 + (endo ~ Z_continuous) + fe(fe1)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # First-stage F-statistic (IID)
    @test Regress.first_stage(m).F_nonrobust[1] ≈ F_IID rtol = RTOL_F_IID

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = Regress.iv(
        Regress.TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous) + fe(fe1)),
        save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = Regress.iv(
        Regress.TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous) + fe(fe1)),
        save_cluster = [
            :fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: IV continuous fe1+fe2" tags = [:iv, :fe, :validation] begin
    using Regress
    using Regress: fe
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using Regress: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6
    RTOL_F_IID = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: x1, x2, fit_endo (no intercept with FE)
    COEF = [0.5061660563758806, 0.30523155864404355, 2.0222299546040468]
    R2 = 0.9216172253226353
    F_IID = 316.751960599266

    # Standard errors by vcov type
    SE_HC1 = [0.04294984682781066, 0.03504025401505687, 0.056889544784780255]
    SE_CLUSTER_FE1 = [0.05006236281851059, 0.0465853775035714, 0.04359836635454813]
    SE_CLUSTER_FE1_FE2 = [0.04054869802568801, 0.04106878430964485, 0.04140086381159971]

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = Regress.iv(Regress.TSLS(), df, @formula(y ~
                                                x1 + x2 + (endo ~ Z_continuous) + fe(fe1) +
                                                fe(fe2)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # First-stage F-statistic (IID)
    @test Regress.first_stage(m).F_nonrobust[1] ≈ F_IID rtol = RTOL_F_IID

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = Regress.iv(
        Regress.TSLS(), df, @formula(y ~
                                     x1 + x2 + (endo ~ Z_continuous) + fe(fe1) + fe(fe2)),
        save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = Regress.iv(
        Regress.TSLS(), df, @formula(y ~
                                     x1 + x2 + (endo ~ Z_continuous) + fe(fe1) + fe(fe2)),
        save_cluster = [:fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: IV categorical no FE" tags = [:iv, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using Regress: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6
    RTOL_F_IID = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: (Intercept), x1, x2, fit_endo
    COEF = [1.4449688608098197, 0.5111036193727284, 0.3184713859664605, 2.028561669636886]
    R2 = 0.7946588621680999
    F_IID = 30.067717609367953

    # Standard errors by vcov type
    # Note: HC1 SE computed with DOF convention that counts intercept (dof_res = n - k)
    SE_HC1 = [
        0.11238654648377033, 0.07279260932749305, 0.05523960188238045, 0.13008646066168578]
    SE_CLUSTER_FE1 = [
        0.18605087076531016, 0.06793250887441085, 0.06010106949621691, 0.10440314279197954]
    SE_CLUSTER_FE1_FE2 = [
        0.3813059794520838, 0.06925420483447764, 0.051594054883199916, 0.09945766983346215]

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = Regress.iv(Regress.TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_cat)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # First-stage F-statistic (IID)
    @test Regress.first_stage(m).F_nonrobust[1] ≈ F_IID rtol = RTOL_F_IID

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = Regress.iv(Regress.TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_cat)), save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = Regress.iv(Regress.TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_cat)), save_cluster = [
        :fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: IV categorical fe1" tags = [:iv, :fe, :validation] begin
    using Regress
    using Regress: fe
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using Regress: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6
    RTOL_F_IID = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: x1, x2, fit_endo (no intercept with FE)
    COEF = [0.5206045994953721, 0.3001680802913261, 2.0223390331838935]
    R2 = 0.8318476170503317
    F_IID = 33.01547709081618

    # Standard errors by vcov type
    SE_HC1 = [0.06956941203475926, 0.04977094038737541, 0.11884473120651287]
    SE_CLUSTER_FE1 = [0.06985333497029948, 0.05641734719583338, 0.08709835505707232]
    SE_CLUSTER_FE1_FE2 = [0.08120122796778467, 0.04237671027164699, 0.13254055303974727]

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = Regress.iv(Regress.TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_cat) + fe(fe1)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # First-stage F-statistic (IID)
    @test Regress.first_stage(m).F_nonrobust[1] ≈ F_IID rtol = RTOL_F_IID

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = Regress.iv(
        Regress.TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_cat) + fe(fe1)), save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = Regress.iv(
        Regress.TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_cat) + fe(fe1)),
        save_cluster = [
            :fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: IV categorical fe1+fe2" tags = [:iv, :fe, :validation] begin
    using Regress
    using Regress: fe
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using Regress: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6
    RTOL_F_IID = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: x1, x2, fit_endo (no intercept with FE)
    COEF = [0.47887536778475115, 0.3035932600900931, 2.0960967017271352]
    R2 = 0.9258190793878247
    F_IID = 32.093732305074198

    # Standard errors by vcov type
    SE_HC1 = [0.047813248315275475, 0.03413860011710209, 0.08038717698779545]
    SE_CLUSTER_FE1 = [0.04961946859843902, 0.04500423266729828, 0.07612596517791004]
    SE_CLUSTER_FE1_FE2 = [0.03729626919528492, 0.03957226570997816, 0.06898871674769344]

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = Regress.iv(Regress.TSLS(), df, @formula(y ~
                                                x1 + x2 + (endo ~ Z_cat) + fe(fe1) +
                                                fe(fe2)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # First-stage F-statistic (IID)
    @test Regress.first_stage(m).F_nonrobust[1] ≈ F_IID rtol = RTOL_F_IID

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = Regress.iv(
        Regress.TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_cat) + fe(fe1) + fe(fe2)),
        save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = Regress.iv(
        Regress.TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_cat) + fe(fe1) + fe(fe2)),
        save_cluster = [
            :fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: IV both no FE" tags = [:iv, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using Regress: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6
    RTOL_F_IID = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: (Intercept), x1, x2, fit_endo
    COEF = [1.4304839335133355, 0.5042848062091939, 0.3179352916746909, 2.0475603520642744]
    R2 = 0.7965074141689706
    F_IID = 103.8605561494657

    # Standard errors by vcov type
    # Note: HC1 SE computed with DOF convention that counts intercept (dof_res = n - k)
    SE_HC1 = [
        0.07681532183851889, 0.06163165142017032, 0.054993238273362964, 0.06861932811803322]
    SE_CLUSTER_FE1 = [
        0.17018634154522, 0.07514028865235683, 0.05963667664288283, 0.06776228338477014]
    SE_CLUSTER_FE1_FE2 = [
        0.409086031649277, 0.07112217630544056, 0.050625432012824316, 0.06370024027894665]

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = Regress.iv(Regress.TSLS(), df, @formula(y ~
                                                x1 + x2 + (endo ~ Z_continuous + Z_cat)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # First-stage F-statistic (IID)
    @test Regress.first_stage(m).F_nonrobust[1] ≈ F_IID rtol = RTOL_F_IID

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = Regress.iv(
        Regress.TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous + Z_cat)),
        save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = Regress.iv(
        Regress.TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous + Z_cat)),
        save_cluster = [
            :fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: IV both fe1" tags = [:iv, :fe, :validation] begin
    using Regress
    using Regress: fe
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using Regress: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6
    RTOL_F_IID = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: x1, x2, fit_endo (no intercept with FE)
    COEF = [0.5166978714883292, 0.2999822484171923, 2.0329668227378943]
    R2 = 0.8324606828041137
    F_IID = 119.21439932462302

    # Standard errors by vcov type
    SE_HC1 = [0.058224459538328215, 0.04982056618001574, 0.06147995587425259]
    SE_CLUSTER_FE1 = [0.07273678884333665, 0.056459015962390516, 0.06259044815925405]
    SE_CLUSTER_FE1_FE2 = [0.06715897511300703, 0.04188121384871312, 0.0655774395827146]

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = Regress.iv(Regress.TSLS(), df, @formula(y ~
                                                x1 + x2 + (endo ~ Z_continuous + Z_cat) +
                                                fe(fe1)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # First-stage F-statistic (IID)
    @test Regress.first_stage(m).F_nonrobust[1] ≈ F_IID rtol = RTOL_F_IID

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = Regress.iv(
        Regress.TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous + Z_cat) + fe(fe1)),
        save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = Regress.iv(
        Regress.TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous + Z_cat) + fe(fe1)),
        save_cluster = [:fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: IV both fe1+fe2" tags = [:iv, :fe, :validation] begin
    using Regress
    using Regress: fe
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using Regress: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6
    RTOL_F_IID = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: x1, x2, fit_endo (no intercept with FE)
    COEF = [0.4965363222169623, 0.3046534721111661, 2.0482944174190525]
    R2 = 0.9232298394541512
    F_IID = 117.462623946516

    # Standard errors by vcov type
    SE_HC1 = [0.0408364626875171, 0.03468144876779309, 0.04425207052902298]
    SE_CLUSTER_FE1 = [0.04675131292387499, 0.0459779939525257, 0.031920503255503524]
    # Note: SE_CLUSTER_FE1_FE2 not available in reference CSV (truncated)

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = Regress.iv(Regress.TSLS(), df,
        @formula(y ~
                 x1 + x2 + (endo ~ Z_continuous + Z_cat) + fe(fe1) +
                 fe(fe2)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # First-stage F-statistic (IID)
    @test Regress.first_stage(m).F_nonrobust[1] ≈ F_IID rtol = RTOL_F_IID

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = Regress.iv(Regress.TSLS(), df,
        @formula(y ~ x1 + x2 + (endo ~ Z_continuous + Z_cat) + fe(fe1) + fe(fe2)),
        save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: inference with cluster-robust standard errors" tags = [
    :ols, :iv, :validation] begin
    # With CR1, p-values, confidence intervals and the Wald F test use G - 1 degrees of
    # freedom (fixest's default ssc(t.df = "min")); with HC1 they use dof_residual.
    # Reference values from R fixest 0.14.1 (test/validation/fixest_validation.R).
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, confint, coeftable, dof_residual
    using Regress: HC1, HC3, CR1, TSLS, fe, inference_dof

    RTOL = 1e-8
    # p-values below 1e-100 count as zero: fixest reports some of them as exactly 0.
    close(a, b) = all(isapprox.(a, b; rtol = RTOL, atol = 1e-100))

    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)

    pvalues(m) = coeftable(m).cols[4]

    # Coefficients x1, x2, endo. Per vcov: t degrees of freedom, p-values, lower and
    # upper 95% bounds, Wald F of all slopes and its p-value.

    # OLS + fe1
    REF_OLS_FE1 = Dict(
        "HC1" => (dof = 977,
            p = [3.48528183353316e-14, 2.87925175407065e-09, 0.0],
            lo = [0.311301157014455, 0.198607005162171, 2.22568969729027],
            hi = [0.524480679608845, 0.391957563520998, 2.37782911047005],
            F = 1386.72715633561, Fp = 0.0),
        "cluster_fe1" => (dof = 19,
            p = [1.07013486731967e-06, 3.99713928556967e-05, 6.61783329236259e-25],
            lo = [0.293560867531862, 0.178891121809268, 2.237092318095],
            hi = [0.542220969091438, 0.4116734468739, 2.36642648966532],
            F = 1868.48165953278, Fp = 1.20121043805327e-23),
        "cluster_fe1_fe2" => (dof = 9,
            p = [4.47479080386939e-05, 8.41611175148278e-05, 3.16603361663839e-12],
            lo = [0.2887218095424, 0.196243989839442, 2.19516587354446],
            hi = [0.547060027080901, 0.394320578843726, 2.40835293421586],
            F = 1609.09258390101, Fp = 1.33733248319332e-12)
    )
    m_ols_fe1 = Regress.ols(df, @formula(y ~ x1 + x2 + endo + fe(fe1)),
        save_cluster = [:fe1, :fe2])

    # IV Z_continuous + fe1
    REF_IV_ZCONT_FE1 = Dict(
        "HC1" => (dof = 977,
            p = [6.50777966216878e-17, 2.60461291170612e-09, 3.2561790677419e-112],
            lo = [0.397920492590315, 0.202059754230213, 1.87696647857667],
            hi = [0.636465175852473, 0.397951830526416, 2.18627419223868],
            F = 428.54914112953, Fp = 1.28219948514126e-177),
        "cluster_fe1" => (dof = 19,
            p = [2.33385267849314e-06, 4.0768592653215e-05, 5.68442410385204e-16],
            lo = [0.35434940687578, 0.181552830356515, 1.86093159841103],
            hi = [0.680036261567008, 0.418458754400113, 2.20230907240432],
            F = 691.080179253012, Fp = 1.44025312181294e-19),
        "cluster_fe1_fe2" => (dof = 9,
            p = [2.60780618882209e-05, 5.1316813783979e-05, 2.44223458635595e-10],
            lo = [0.367907320166136, 0.205632407537045, 1.87871445256877],
            hi = [0.666478348276652, 0.394379177219584, 2.18452621824658],
            F = 6907.70123169775, Fp = 1.91380865926116e-15)
    )
    m_iv_Zcont_fe1 = Regress.iv(
        TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous) + fe(fe1)),
        save_cluster = [:fe1, :fe2])

    estimators = Dict("HC1" => HC1(), "cluster_fe1" => CR1(:fe1),
        "cluster_fe1_fe2" => CR1(:fe1, :fe2))
    for (m, ref) in ((m_ols_fe1, REF_OLS_FE1), (m_iv_Zcont_fe1, REF_IV_ZCONT_FE1))
        names = replace.(coefnames(m), "fit_" => "")
        idx = [findfirst(==(c), names) for c in ("x1", "x2", "endo")]
        for (vt, k) in estimators
            r = ref[vt]
            mk = m + vcov(k)
            ci = confint(mk)
            @test inference_dof(mk) == r.dof
            @test close(pvalues(mk)[idx], r.p)
            @test ci[idx, 1] ≈ r.lo rtol = RTOL
            @test ci[idx, 2] ≈ r.hi rtol = RTOL
            @test mk.F ≈ r.F rtol = RTOL
            @test close(mk.p, r.Fp)
            # The interval computed directly from the estimator agrees.
            @test confint(k, m)[idx, :] ≈ ci[idx, :] rtol = RTOL
        end
    end

    # G - 1 is the smallest number of clusters minus one; non-cluster estimators keep
    # dof_residual, and dof_residual itself does not change.
    m = m_ols_fe1
    @test inference_dof(m + vcov(CR1(:fe1))) == 19
    @test inference_dof(m + vcov(CR1(:fe1, :fe2))) == 9
    @test inference_dof(m + vcov(HC3())) == dof_residual(m)
    @test dof_residual(m + vcov(CR1(:fe1))) == dof_residual(m)

    # The printed summary states the degrees of freedom when they are G - 1.
    @test occursin("t and F tests with G - 1 = 19 dof", sprint(show, m + vcov(CR1(:fe1))))
    @test !occursin("G - 1", sprint(show, m + vcov(HC1())))
end
