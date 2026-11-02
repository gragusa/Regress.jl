@testitem "Binary weighted least squares reuses workspace buffers" tags = [:glmfe, :probit] begin
    using Regress, LinearAlgebra
    using Regress: GLMWorkspace, wls_absorbed!

    function solve_with_workspace!(beta, X, z, h, basis, ws)
        return wls_absorbed!(beta, X, z, h, basis;
            hz = ws.resid, hX = ws.hX, XtWX = ws.XtWX, XtWz = ws.XtWz)
    end
    function allocated_solve!(beta, X, z, h, basis, ws)
        solve_with_workspace!(beta, X, z, h, basis, ws)
        return @allocated solve_with_workspace!(beta, X, z, h, basis, ws)
    end

    for T in (Float32, Float64)
        x = T.(range(-1, 1; length = 256))
        X = hcat(ones(T, length(x)), x, x .^ 2)
        z = T(0.2) .+ T(0.7) .* x .- T(0.3) .* x .^ 2 .+ T(0.1) .* sin.(T(3) .* x)
        h = T.(range(0.2, 2; length = length(x)))
        beta = zeros(T, size(X, 2))
        ws = GLMWorkspace(X)
        saved_X, saved_z = copy(X), copy(z)
        tolerance = T === Float32 ? 1e-5 : 1e-11
        for basis in (trues(3), BitVector([false, true, true]), falses(3), trues(3))
            h .*= T(1.1)
            saved_h = copy(h)
            expected = zeros(T, length(beta))
            if any(basis)
                expected[basis] = (sqrt.(h) .* X[:, basis]) \ (sqrt.(h) .* z)
            end
            @test solve_with_workspace!(beta, view(X, :, :), view(z, :), h, basis, ws) ===
                  beta
            @test beta ≈ expected atol=tolerance
            @test X == saved_X
            @test z == saved_z
            @test h == saved_h
        end
        @test allocated_solve!(beta, X, z, h, trues(3), ws) < 1024
        @test wls_absorbed!(beta, X, z, h, trues(3)) ≈
              (sqrt.(h) .* X) \ (sqrt.(h) .* z) atol=tolerance
        @test_throws "matching row axes" solve_with_workspace!(
            beta, X, z[1:(end - 1)], h, trues(3), ws)
        @test_throws "matching column axes" solve_with_workspace!(
            beta, X, z, h, trues(2), ws)
        @test_throws "weighted design buffer" wls_absorbed!(beta, X, z, h, trues(3);
            hz = ws.resid, hX = similar(X, size(X, 1), 4),
            XtWX = ws.XtWX, XtWz = ws.XtWz)
        @test_throws "matrix buffer" wls_absorbed!(beta, X, z, h, trues(3);
            hz = ws.resid, hX = ws.hX, XtWX = zeros(T, 2, 2), XtWz = ws.XtWz)
    end
end
