##############################################################################
##
## Tests for FE-Based Instruments (Experimental Feature)
##
## This tests that using fe() inside the instrument specification produces
## the same results as the standard dummy expansion approach.
##
##############################################################################

using Regress, CategoricalArrays, DataFrames, Test, StatsBase, StableRNGs
using StatsAPI: coef, stderror, vcov, confint, nobs, dof_residual, r2

@testset "FE-based instruments" begin

    ##########################################################################
    ## Setup: Create test data with categorical variables
    ##########################################################################

    # Create a dataset with categorical instruments
    n = 5000
    rng = StableRNG(42)

    # Categorical variables for FE instruments
    group1 = repeat(1:50, inner = n÷50)   # 50 groups
    group2 = repeat(1:10, outer = n÷10)   # 10 groups
    quarter = mod1.(1:n, 4)              # 4 quarters

    # Continuous variables
    x1 = randn(rng, n)

    # Endogenous variable correlated with error
    # Instruments: group1×quarter and group2×quarter interactions
    z_effect = zeros(n)
    for i in 1:n
        # Group means that vary by group×quarter
        z_effect[i] = 0.1 * group1[i] + 0.05 * group2[i] + 0.02 * quarter[i] +
                      0.01 * group1[i] * quarter[i] + 0.005 * group2[i] * quarter[i]
    end

    error = randn(rng, n)
    endo = z_effect + 0.5 * error + randn(rng, n)  # Endogenous (correlated with error)

    # Outcome
    y = 1.0 .+ 2.0 .* x1 .+ 3.0 .* endo .+ error

    df = DataFrame(
        y = y,
        x1 = x1,
        endo = endo,
        group1 = categorical(group1),
        group2 = categorical(group2),
        quarter = categorical(quarter)
    )

    ##########################################################################
    ## Test 1: Basic FE instruments vs standard instruments
    ##########################################################################

    @testset "Basic equivalence" begin
        # Standard approach: explicit dummy expansion
        model_std = iv(TSLS(), df,
            @formula(y ~ x1 + (endo ~ group1&quarter) + fe(group1)))

        # FE-based approach: use fe() in instruments
        model_fe = iv(TSLS(), df,
            @formula(y ~ x1 + (endo ~ fe(group1)&fe(quarter)) + fe(group1)))

        # Coefficients should match
        @test coef(model_std) ≈ coef(model_fe) rtol=1e-4

        # Standard errors should match
        @test stderror(model_std) ≈ stderror(model_fe) rtol=1e-4

        # Vcov matrices should match
        @test vcov(model_std) ≈ vcov(model_fe) rtol=1e-4

        # Other statistics should match
        @test r2(model_std) ≈ r2(model_fe) rtol=1e-4
        @test nobs(model_std) == nobs(model_fe)
        @test dof_residual(model_std) == dof_residual(model_fe)

        # Residuals should match
        @test model_std.rss ≈ model_fe.rss rtol=1e-4
    end

    ##########################################################################
    ## Test 2: Multiple FE instrument interactions
    ##########################################################################

    @testset "Multiple FE interactions" begin
        # Standard approach with two interaction terms
        model_std = iv(TSLS(), df,
            @formula(y ~
                     x1 + (endo ~ group1&quarter + group2&quarter) + fe(group1) +
                     fe(group2)))

        # FE-based approach
        model_fe = iv(TSLS(), df,
            @formula(y ~
                     x1 + (endo ~ fe(group1)&fe(quarter) + fe(group2)&fe(quarter)) +
                     fe(group1) + fe(group2)))

        # Coefficients should match
        @test coef(model_std) ≈ coef(model_fe) rtol=1e-4

        # Standard errors should match
        @test stderror(model_std) ≈ stderror(model_fe) rtol=1e-4

        # R-squared should match
        @test r2(model_std) ≈ r2(model_fe) rtol=1e-4
    end

    ##########################################################################
    ## Test 3: With robust standard errors (+ vcov)
    ##########################################################################

    @testset "Robust standard errors" begin
        model_std = iv(TSLS(), df,
            @formula(y ~ x1 + (endo ~ group1&quarter) + fe(group1)))
        model_fe = iv(TSLS(), df,
            @formula(y ~ x1 + (endo ~ fe(group1)&fe(quarter)) + fe(group1)))

        # Apply HC3 robust standard errors
        model_std_hc3 = model_std + vcov(HC3())
        model_fe_hc3 = model_fe + vcov(HC3())

        # Coefficients should still match
        @test coef(model_std_hc3) ≈ coef(model_fe_hc3) rtol=1e-4

        # HC3 standard errors should match
        @test stderror(model_std_hc3) ≈ stderror(model_fe_hc3) rtol=1e-4

        # HC3 vcov should match
        @test vcov(model_std_hc3) ≈ vcov(model_fe_hc3) rtol=1e-4
    end

    ##########################################################################
    ## Test 4: Performance - FE approach should be faster
    ##########################################################################

    @testset "Performance improvement" begin
        # This is a smoke test - just ensure both run without error
        # and the FE approach uses less memory

        # Standard approach
        t_std = @elapsed model_std = iv(TSLS(), df,
            @formula(y ~
                     x1 + (endo ~ group1&quarter + group2&quarter) + fe(group1) +
                     fe(group2)))

        # FE-based approach (should be faster for large categorical instruments)
        t_fe = @elapsed model_fe = iv(TSLS(), df,
            @formula(y ~
                     x1 + (endo ~ fe(group1)&fe(quarter) + fe(group2)&fe(quarter)) +
                     fe(group1) + fe(group2)))

        # Both should produce valid results
        @test all(isfinite.(coef(model_std)))
        @test all(isfinite.(coef(model_fe)))

        # Note: For this small test case, timing may not show improvement
        # The real benefit is with large categorical variables (100s of levels)
    end

    ##########################################################################
    ## Test 5: Confidence intervals
    ##########################################################################

    @testset "Confidence intervals" begin
        model_std = iv(TSLS(), df,
            @formula(y ~ x1 + (endo ~ group1&quarter) + fe(group1)))
        model_fe = iv(TSLS(), df,
            @formula(y ~ x1 + (endo ~ fe(group1)&fe(quarter)) + fe(group1)))

        ci_std = confint(model_std)
        ci_fe = confint(model_fe)

        # Confidence intervals should match
        @test ci_std ≈ ci_fe rtol=1e-4
    end

    ##########################################################################
    ## Test 6: F-statistic and p-values
    ##########################################################################

    @testset "Model F-statistics" begin
        model_std = iv(TSLS(), df,
            @formula(y ~ x1 + (endo ~ group1&quarter) + fe(group1)))
        model_fe = iv(TSLS(), df,
            @formula(y ~ x1 + (endo ~ fe(group1)&fe(quarter)) + fe(group1)))

        # Model F-statistics should match
        @test model_std.F ≈ model_fe.F rtol=1e-4
        @test model_std.p ≈ model_fe.p rtol=1e-4
    end
end
