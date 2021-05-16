module PolynomialBasis

# using StaticArrays
using LinearAlgebra
import DynamicPolynomials
import StaticPolynomials

DP = DynamicPolynomials
SP = StaticPolynomials

include("lagrange_polynomials.jl")
include("hermite_polynomials.jl")
include("lagrange_basis.jl")
include("hermite_basis.jl")
include("tensor_product_basis.jl")
include("interpolation.jl")

export LagrangePolynomialBasis, TensorProductBasis, gradient, hessian,
        InterpolatingPolynomial, update!

end # module
