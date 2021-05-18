using Test
using DynamicPolynomials
# using Revise
using PolynomialBasis
PB = PolynomialBasis

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

basis = PB.HermiteTensorProductBasis(2)

testvals = zeros(16)
testvals[1] = 1.0
@test allapprox(basis(-1.,-1.),testvals)

fill!(testvals,0.0)
testvals[5] = 1.0
@test allapprox(basis(-1.,1.),testvals)

fill!(testvals,0.0)
testvals[9] = 1.0
@test allapprox(basis(1,-1),testvals)

fill!(testvals,0.0)
testvals[13] = 1.0
@test allapprox(basis(1,1),testvals)

testvals = zeros(16,2)
testvals[2,2] = 1.0
testvals[3,1] = 1.0
@test allapprox(gradient(basis,-1,-1),testvals)

fill!(testvals,0.0)
testvals[6,2] = 1.0
testvals[7,1] = 1.0
@test allapprox(gradient(basis,-1,1),testvals)

fill!(testvals,0.0)
testvals[10,2] = 1.0
testvals[11,1] = 1.0
@test allapprox(gradient(basis,1,-1),testvals)

fill!(testvals,0.0)
testvals[14,2] = 1.0
testvals[15,1] = 1.0
@test allapprox(gradient(basis,1,1),testvals)
