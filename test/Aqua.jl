using Test
using Aqua
using Regress

@testset "Aqua.jl" begin
    Aqua.test_all(Regress)
end
