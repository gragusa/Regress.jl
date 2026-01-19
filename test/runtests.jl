using Regress, Test

@testset "Aqua" begin
    include("Aqua.jl")
end
@testset "formula" begin
    include("formula.jl")
end
@testset "fit" begin
    include("fit.jl")
end
@testset "predict" begin
    include("predict.jl")
end
@testset "partial out" begin
    include("partial_out.jl")
end
@testset "collinearity" begin
    include("collinearity.jl")
end
@testset "model + vcov" begin
    include("model_plus_vcov.jl")
end
@testset "K-class estimators" begin
    include("kclass.jl")
end
