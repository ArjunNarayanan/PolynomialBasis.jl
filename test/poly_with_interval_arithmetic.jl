using IntervalArithmetic
# using Revise
using PolynomialBasis

basis = TensorProductBasis(2,2)
poly = InterpolatingPolynomial(1,basis)
box = IntervalBox(-1..1,2)
update!(poly,rand(9))
