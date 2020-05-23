module PolynomialBasis

using StaticArrays
import DynamicPolynomials
import StaticPolynomials

DP = DynamicPolynomials
SP = StaticPolynomials

include("lagrange_polynomials.jl")
include("lagrange_basis.jl")

end # module
