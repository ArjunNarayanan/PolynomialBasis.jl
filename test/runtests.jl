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

@safetestset "Test Hermite Basis" begin
    include("test_hermite_basis.jl")
end

@safetestset "Test Tensor Product Basis" begin
    include("test_lagrange_tensor_product_basis.jl")
end

@safetestset "Test Hermite Tensor Product Basis" begin
    include("test_hermite_tensor_product_basis.jl")
end

@safetestset "Test Lagrange Interpolating Polynomial" begin
    include("test_lagrange_interpolation.jl")
end

@safetestset "Test Hermite Interpolating Polynomial" begin
    include("test_hermite_interpolation.jl")
end
