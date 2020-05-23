using Test
import DynamicPolynomials: @polyvar
# import StaticArrays
# using Revise
using PolynomialBasis

PB = PolynomialBasis

@polyvar x

function allequal(v1,v2)
    return all(v1 .≈ v2)
end

funcs = [2.0+0.0x]
deleteat!(funcs,1)
@test_throws AssertionError PB.LagrangePolynomialBasis(funcs,[])

funcs = [2.0+0.0x]
@test_throws AssertionError PB.LagrangePolynomialBasis(funcs,[0.0])

funcs = [1.0+0.0x]
@test_throws AssertionError PB.LagrangePolynomialBasis(funcs,[0.0,1.0])

@polyvar y
funcs = [1.0+x+y,-x-y]
@test_throws AssertionError PB.LagrangePolynomialBasis(funcs,[-1.0,1.0])

funcs = [x^2,1.0+0.0x,-x^2]
@test_throws AssertionError PB.LagrangePolynomialBasis(funcs,[-1.0,0.0,1.0])

@test_throws AssertionError PB.LagrangePolynomialBasis(-1)

basis = PB.LagrangePolynomialBasis(0)
@test allequal(basis.points,[0.0])
@test allequal(basis(-1.0),[1.0])
@test allequal(basis(1.0),[1.0])
@test allequal(PB.derivative(basis,-1.0),[0.0])
@test allequal(PB.derivative(basis,1.0),[0.0])

basis = PB.LagrangePolynomialBasis(1)
@test allequal(basis.points,[-1.0 1.0])
@test allequal(basis(-1.0),[1.0,0.0])
@test allequal(basis(1.0),[0.0,1.0])
@test allequal(PB.derivative(basis,-1.0),[-0.5,0.5])
@test allequal(PB.derivative(basis,1.0),[-0.5,0.5])

basis = PB.LagrangePolynomialBasis(2)
@test allequal(basis.points,[-1.0 0.0 1.0])
