# Validation tests comparing Regress.jl with R fixest package
#
# This test suite validates OLS and IV estimation with:
# - Fixed effects (none, fe1, fe1+fe2)
# - Variance estimators (HC1, cluster_fe1, two-way cluster)
# - Continuous and categorical instruments
#
# Reference values are hardcoded from R fixest (see fixest_validation.R for methodology).
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
    using CovarianceMatrices: HC1, CR1

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
    m = ols(df, @formula(y ~ x1 + x2 + endo))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = ols(df, @formula(y ~ x1 + x2 + endo), save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = ols(df, @formula(y ~ x1 + x2 + endo), save_cluster = [:fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: OLS with fe1" tags = [:ols, :fe, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using CovarianceMatrices: HC1, CR1

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
    m = ols(df, @formula(y ~ x1 + x2 + endo + fe(fe1)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = ols(df, @formula(y ~ x1 + x2 + endo + fe(fe1)), save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = ols(df, @formula(y ~ x1 + x2 + endo + fe(fe1)), save_cluster = [:fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: OLS with fe1+fe2" tags = [:ols, :fe, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using CovarianceMatrices: HC1, CR1

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
    SE_HC1 = [0.035059220996074746, 0.032902861007155934, 0.026088733299502308]
    SE_CLUSTER_FE1 = [0.03795362647742814, 0.0419840271459015, 0.01828786423527352]
    SE_CLUSTER_FE1_FE2 = [0.03311809661530514, 0.04098378676772609, 0.01906479240456021]

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = ols(df, @formula(y ~ x1 + x2 + endo + fe(fe1) + fe(fe2)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = ols(df, @formula(y ~ x1 + x2 + endo + fe(fe1) + fe(fe2)), save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = ols(df, @formula(y ~ x1 + x2 + endo + fe(fe1) + fe(fe2)), save_cluster = [
        :fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: IV continuous no FE" tags = [:iv, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using CovarianceMatrices: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6
    RTOL_F_KP = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: (Intercept), x1, x2, fit_endo
    COEF = [1.426078709850737, 0.5022110369396626, 0.31777225217776084, 2.0533383199942645]
    R2 = 0.7970527457662797
    F_KP = 135.55368890700018

    # Standard errors by vcov type
    # Note: HC1 SE computed with DOF convention that counts intercept (dof_res = n - k)
    SE_HC1 = [
        0.08893165980839529, 0.06513596685507532, 0.05498467927980058, 0.08847046479733618]
    SE_CLUSTER_FE1 = [
        0.17123687353987066, 0.08144868937473768, 0.05950490558096538, 0.07952747899994171]
    SE_CLUSTER_FE1_FE2 = [
        0.42545555594227763, 0.0746658815153094, 0.0501996102660622, 0.07444197352128866]

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # First-stage F-statistic (Kleibergen-Paap)
    @test m.F_kp ≈ F_KP rtol = RTOL_F_KP

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous)), save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous)), save_cluster = [
        :fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: IV continuous fe1" tags = [:iv, :fe, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using CovarianceMatrices: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6
    RTOL_F_KP = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: x1, x2, fit_endo (no intercept with FE)
    COEF = [0.517192834221396, 0.3000057923783143, 2.0316203354076725]
    R2 = 0.832384325466159
    F_KP = 145.51765918382873

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
    m = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous) + fe(fe1)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # First-stage F-statistic (Kleibergen-Paap)
    @test m.F_kp ≈ F_KP rtol = RTOL_F_KP

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous) + fe(fe1)), save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous) + fe(fe1)), save_cluster = [
        :fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: IV continuous fe1+fe2" tags = [:iv, :fe, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using CovarianceMatrices: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6
    RTOL_F_KP = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: x1, x2, fit_endo (no intercept with FE)
    COEF = [0.5061660563758806, 0.30523155864404355, 2.0222299546040468]
    R2 = 0.9216172253226353
    F_KP = 145.19326049090716

    # Standard errors by vcov type
    SE_HC1 = [0.04297204886952502, 0.035058367355121675, 0.056918952667222356]
    SE_CLUSTER_FE1 = [0.05008774297873092, 0.04660899493348077, 0.043620469456781476]
    SE_CLUSTER_FE1_FE2 = [0.04054869802568801, 0.04106878430964485, 0.04140086381159971]

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous) + fe(fe1) + fe(fe2)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # First-stage F-statistic (Kleibergen-Paap)
    @test m.F_kp ≈ F_KP rtol = RTOL_F_KP

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = iv(
        TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous) + fe(fe1) + fe(fe2)), save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = iv(
        TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous) + fe(fe1) + fe(fe2)),
        save_cluster = [:fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: IV categorical no FE" tags = [:iv, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using CovarianceMatrices: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6
    RTOL_F_KP = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: (Intercept), x1, x2, fit_endo
    COEF = [1.4449688608098197, 0.5111036193727284, 0.3184713859664605, 2.028561669636886]
    R2 = 0.7946588621680999
    F_KP = 24.35925856663632

    # Standard errors by vcov type
    # Note: HC1 SE computed with DOF convention that counts intercept (dof_res = n - k)
    SE_HC1 = [
        0.11238654648377033, 0.07279260932749305, 0.05523960188238045, 0.13008646066168578]
    SE_CLUSTER_FE1 = [
        0.1859575420056042, 0.0678984318675643, 0.060070920976889805, 0.10435077100900533]
    SE_CLUSTER_FE1_FE2 = [
        0.3811147048077682, 0.06921946482485047, 0.05156817374042106, 0.09940777884971522]

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_cat)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # First-stage F-statistic (Kleibergen-Paap)
    @test m.F_kp ≈ F_KP rtol = RTOL_F_KP

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_cat)), save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_cat)), save_cluster = [
        :fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: IV categorical fe1" tags = [:iv, :fe, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using CovarianceMatrices: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6
    RTOL_F_KP = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: x1, x2, fit_endo (no intercept with FE)
    COEF = [0.5206045994953721, 0.3001680802913261, 2.0223390331838935]
    R2 = 0.8318476170503317
    F_KP = 25.903487111101313

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
    m = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_cat) + fe(fe1)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # First-stage F-statistic (Kleibergen-Paap)
    @test m.F_kp ≈ F_KP rtol = RTOL_F_KP

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_cat) + fe(fe1)), save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_cat) + fe(fe1)), save_cluster = [
        :fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: IV categorical fe1+fe2" tags = [:iv, :fe, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using CovarianceMatrices: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6
    RTOL_F_KP = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: x1, x2, fit_endo (no intercept with FE)
    COEF = [0.47887536778475115, 0.3035932600900931, 2.0960967017271352]
    R2 = 0.9258190793878247
    F_KP = 24.896281475208674

    # Standard errors by vcov type
    SE_HC1 = [0.047837964392560675, 0.034156247365691846, 0.08042873148888426]
    SE_CLUSTER_FE1 = [0.04964462422418577, 0.04502704850281556, 0.07616455882559972]
    SE_CLUSTER_FE1_FE2 = [0.03729626919528492, 0.03957226570997816, 0.06898871674769344]

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_cat) + fe(fe1) + fe(fe2)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # First-stage F-statistic (Kleibergen-Paap)
    @test m.F_kp ≈ F_KP rtol = RTOL_F_KP

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_cat) + fe(fe1) + fe(fe2)), save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = iv(
        TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_cat) + fe(fe1) + fe(fe2)), save_cluster = [
            :fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: IV both no FE" tags = [:iv, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using CovarianceMatrices: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6
    RTOL_F_KP = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: (Intercept), x1, x2, fit_endo
    COEF = [1.4304839335133355, 0.5042848062091939, 0.3179352916746909, 2.0475603520642744]
    R2 = 0.7965074141689706
    F_KP = 41.46776399951962

    # Standard errors by vcov type
    # Note: HC1 SE computed with DOF convention that counts intercept (dof_res = n - k)
    SE_HC1 = [
        0.07681532183851889, 0.06163165142017032, 0.054993238273362964, 0.06861932811803322]
    SE_CLUSTER_FE1 = [
        0.1701009709145438, 0.07510259600456129, 0.059606761077096296, 0.06772829176819219]
    SE_CLUSTER_FE1_FE2 = [
        0.4088808216882091, 0.07108649926466752, 0.05060003676065557, 0.0636682863063922]

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous + Z_cat)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # First-stage F-statistic (Kleibergen-Paap)
    @test m.F_kp ≈ F_KP rtol = RTOL_F_KP

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous + Z_cat)), save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous + Z_cat)), save_cluster = [
        :fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: IV both fe1" tags = [:iv, :fe, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using CovarianceMatrices: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6
    RTOL_F_KP = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: x1, x2, fit_endo (no intercept with FE)
    COEF = [0.5166978714883292, 0.2999822484171923, 2.0329668227378943]
    R2 = 0.8324606828041137
    F_KP = 43.486936945756945

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
    m = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous + Z_cat) + fe(fe1)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # First-stage F-statistic (Kleibergen-Paap)
    @test m.F_kp ≈ F_KP rtol = RTOL_F_KP

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous + Z_cat) + fe(fe1)), save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER

    # Standard errors - cluster(fe1, fe2)
    m_cl2 = iv(TSLS(), df, @formula(y ~ x1 + x2 + (endo ~ Z_continuous + Z_cat) + fe(fe1)),
        save_cluster = [:fe1, :fe2])
    @test stderror(CR1(:fe1, :fe2), m_cl2) ≈ SE_CLUSTER_FE1_FE2 rtol = RTOL_SE_CLUSTER
end

@testitem "fixest: IV both fe1+fe2" tags = [:iv, :fe, :validation] begin
    using Regress
    using DataFrames, CSV, CategoricalArrays
    using StatsBase: coef, stderror, r2
    using CovarianceMatrices: HC1, CR1

    # Tolerances
    RTOL_COEF = 1e-6
    RTOL_SE_HC = 1e-6
    RTOL_SE_CLUSTER = 1e-6
    RTOL_R2 = 1e-6
    RTOL_F_KP = 1e-6

    # ----- Reference values from R fixest -----
    # Coefficients: x1, x2, fit_endo (no intercept with FE)
    COEF = [0.4965363222169623, 0.3046534721111661, 2.0482944174190525]
    R2 = 0.9232298394541512
    F_KP = 43.43219369557059

    # Standard errors by vcov type
    SE_HC1 = [0.04085757225868013, 0.034699376630850265, 0.044274945728549696]
    SE_CLUSTER_FE1 = [0.04677501447821067, 0.04600130345666063, 0.03193668602118458]
    # Note: SE_CLUSTER_FE1_FE2 not available in reference CSV (truncated)

    # ----- Load data -----
    df = DataFrame(CSV.File(joinpath(@__DIR__, "validation", "fixest_validation_data.csv")))
    df.fe1 = categorical(df.fe1)
    df.fe2 = categorical(df.fe2)
    df.Z_cat = categorical(df.Z_cat)

    # ----- Tests -----
    m = iv(TSLS(), df, @formula(y ~
                                x1 + x2 + (endo ~ Z_continuous + Z_cat) + fe(fe1) +
                                fe(fe2)))

    # Coefficients
    @test coef(m) ≈ COEF rtol = RTOL_COEF

    # R-squared
    @test r2(m) ≈ R2 rtol = RTOL_R2

    # First-stage F-statistic (Kleibergen-Paap)
    @test m.F_kp ≈ F_KP rtol = RTOL_F_KP

    # Standard errors - HC1
    @test stderror(HC1(), m) ≈ SE_HC1 rtol = RTOL_SE_HC

    # Standard errors - cluster(fe1)
    m_cl1 = iv(TSLS(), df,
        @formula(y ~ x1 + x2 + (endo ~ Z_continuous + Z_cat) + fe(fe1) + fe(fe2)),
        save_cluster = :fe1)
    @test stderror(CR1(:fe1), m_cl1) ≈ SE_CLUSTER_FE1 rtol = RTOL_SE_CLUSTER
end
