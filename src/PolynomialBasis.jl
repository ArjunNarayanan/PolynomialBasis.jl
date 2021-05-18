module PolynomialBasis

# using StaticArrays
using LinearAlgebra
import DynamicPolynomials
import StaticPolynomials

DP = DynamicPolynomials
SP = StaticPolynomials

include("lagrange_polynomials.jl")
include("hermite_polynomials.jl")
include("common_definitions.jl")
include("lagrange_basis.jl")
include("hermite_basis.jl")
include("lagrange_tensor_product_basis.jl")
# include("hermite_tensor_product_basis.jl")
# include("interpolation.jl")

export LagrangePolynomialBasis, TensorProductBasis, gradient, hessian,
        InterpolatingPolynomial, HermiteTensorProductBasis, update!

end # module
