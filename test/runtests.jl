using SafeTestsets

@safetestset "Test Lagrange Polynomial Construction" begin
    include("test_lagrange_polynomials.jl")
end

@safetestset "Test Hermite Polynomial Construction" begin
    include("test_hermite_polynomials.jl")
end

@safetestset "Test Lagrange Basis" begin
    include("test_lagrange_basis.jl")
end

@safetestset "Test Tensor Product Basis" begin
    include("test_tensor_product_basis.jl")
end

@safetestset "Test Interpolation" begin
    include("test_interpolation.jl")
end
