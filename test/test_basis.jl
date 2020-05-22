using Test
import DynamicPolynomials: @polyvar
import StaticPolynomials
using Revise
using PolynomialBasis

@polyvar x

funcs = [2.0+0.0x]
@test_throws AssertionError PolynomialBasis.LagrangePolynomialBasis(funcs,[0.0])

funcs = [1.0+0.0x]
@test_throws AssertionError PolynomialBasis.LagrangePolynomialBasis(funcs,[0.0,1.0])

points = [0.0]
polys = PolynomialBasis.lagrange_polynomials(x,points)
basis = PolynomialBasis.LagrangePolynomialBasis(polys,points)

points = [-1.0,1.0]
polys = PolynomialBasis.lagrange_polynomials(x,points)
basis = PolynomialBasis.LagrangePolynomialBasis(polys,points)
