module PolynomialBasis

using StaticArrays
import DynamicPolynomials
import StaticPolynomials

DP = DynamicPolynomials
SP = StaticPolynomials

include("lagrange_polynomials.jl")
include("lagrange_basis.jl")
include("tensor_product_basis.jl")
include("interpolation.jl")

export LagrangePolynomialBasis, TensorProductBasis, gradient,
        InterpolatingPolynomial, update!

end # module
