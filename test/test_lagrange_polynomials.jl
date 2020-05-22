using Test
using Revise
using PolynomialBasis
import DynamicPolynomials: @polyvar

function allequal(v1,v2)
    return all(v1 .≈ v2)
end

points = [-1.0,1.0]
matrix = PolynomialBasis.vandermonde_matrix(points)
testmatrix = [1.0  -1.0
              1.0   1.0]
@test allequal(matrix,testmatrix)

points = [-1.0,0.0,1.0]
matrix = PolynomialBasis.vandermonde_matrix(points)
testmatrix = [1.0  -1.0   1.0
              1.0   0.0   0.0
              1.0   1.0   1.0]
@test allequal(matrix,testmatrix)

points = [0.0]
matrix = PolynomialBasis.vandermonde_matrix(points)
testmatrix = [1.0]
@test allequal(matrix,testmatrix)

coeffs = PolynomialBasis.lagrange_polynomial_coefficients([0.0])
testcoeffs = [1.0]
@test allequal(coeffs,testcoeffs)

coeffs = PolynomialBasis.lagrange_polynomial_coefficients([-1.0,1.0])
testcoeffs = [0.5  0.5
             -0.5  0.5]
@test allequal(coeffs,testcoeffs)

coeffs = PolynomialBasis.lagrange_polynomial_coefficients([-1.0,0.0,1.0])
testcoeffs = [+0.0   1.0   0.0
              -0.5   0.0   0.5
              +0.5  -1.0   0.5]
@test allequal(coeffs,testcoeffs)

@polyvar x
coeffs = [1.0]
poly = PolynomialBasis.polynomial_from_coefficients(x,coeffs)
powers = sort!(vcat(poly.x.Z...))
testpowers = [0]
@test allequal(powers,testpowers)
@test allequal(coeffs,sort(poly.a))

coeffs = [1.0,2.0]
poly = PolynomialBasis.polynomial_from_coefficients(x,coeffs)
powers = sort!(vcat(poly.x.Z...))
testpowers = [0,1]
@test allequal(powers,testpowers)
@test allequal(coeffs,sort(poly.a))

polys = PolynomialBasis.lagrange_polynomials(x,[0.0])
@test length(polys) == 1
@test polys[1] == 1.0

polys = PolynomialBasis.lagrange_polynomials(x,[-1.0,1.0])
@test length(polys) == 2
p1 = 0.5 - 0.5x
p2 = 0.5 + 0.5x
@test allequal([p1,p2],polys)

polys = PolynomialBasis.lagrange_polynomials(x,[-1.0,0.0,1.0])
@test length(polys) == 3
p1 = -0.5x + 0.5x^2
p2 = 1.0 - x^2
p3 = 0.5x + 0.5x^2
@test allequal([p1,p2,p3],polys)
