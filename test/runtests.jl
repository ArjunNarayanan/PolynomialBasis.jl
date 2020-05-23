using SafeTestsets

@safetestset "Test Lagrange Polynomial Construction" begin
    include("test_lagrange_polynomials.jl")
end

@safetestset "Test Lagrange Basis" begin
    include("test_lagrange_basis.jl")
end
