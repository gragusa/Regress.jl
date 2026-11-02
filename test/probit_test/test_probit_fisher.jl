@testitem "Probit Fisher weights and working response" tags = [:probit] begin
    using Regress, DataFrames
    using Regress: fit_probit, GLMWorkspace, BinaryResponse, _glm_irls!
    using StatsAPI: deviance

    df = DataFrame(x = repeat([-1.0, 0.0, 1.0], inner = 4),
        y = [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    m = fit_probit(df, @formula(y ~ x), nothing, 100, 1e-10)
    @test deviance(m) ≈ deviance(m.rr)
    fill!(m.pp.beta, 0.0)
    m.rr = BinaryResponse(df.y, m.pp, :y)
    @test m.rr.mu ≈ fill(0.5, length(df.y))
    @test m.rr.one_minus_mu ≈ fill(0.5, length(df.y))
    @test m.rr.resid ≈ df.y .- 0.5
    @test m.rr.dmu_deta ≈ fill(inv(sqrt(2pi)), length(df.y))
    @test m.rr.deviance_contributions ≈ fill(2log(2), length(df.y))
    ws = GLMWorkspace(m.pp.X)
    @test ws isa GLMWorkspace

    @test_throws "Fisher scoring did not converge in 1 iterations" _glm_irls!(
        m, ws, nothing, false, 1, 1e-10)

    # At eta = 0, mu = 1/2 and phi(eta) = 1/sqrt(2pi).
    derivative = inv(sqrt(2pi))
    @test ws.g ≈ df.y .- 0.5
    @test ws.d ≈ fill(derivative, length(df.y))
    @test ws.h ≈ fill(2 / pi, length(df.y))
    @test ws.z ≈ (df.y .- 0.5) ./ derivative

    accepted_deviance = m.rr.deviance
    candidate_deviance = Regress.refresh_response!(
        m.rr, m.pp.X, zeros(size(m.pp.X, 2)), fill(0.5, length(df.y)))
    @test m.rr.eta ≈ fill(0.5, length(df.y))
    @test m.rr.mu ≈ fill(Regress.normcdf(0.5), length(df.y))
    @test m.rr.one_minus_mu ≈ fill(Regress.normcdf(-0.5), length(df.y))
    @test m.rr.resid ≈ m.rr.y .- m.rr.mu
    @test m.rr.dmu_deta ≈ fill(exp(-0.5^2 / 2) / sqrt(2pi), length(df.y))
    @test deviance(m.rr) ≈ candidate_deviance
    @test candidate_deviance ≈ sum(m.rr.deviance_contributions)
    @test m.rr.deviance == accepted_deviance
end
