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
@test_throws AssertionError PB.LagrangePolynomialBasis(1,start=2)
@test_throws AssertionError PB.LagrangePolynomialBasis(2,stop=-2.0)

basis = PB.LagrangePolynomialBasis(0)
@test allequal(basis.points,[0.0])
@test allequal(basis(-1.0),[1.0])
@test allequal(basis(1.0),[1.0])
@test allequal(PB.derivative(basis,-1.0),[0.0])
@test allequal(PB.derivative(basis,1.0),[0.0])

basis = PB.LagrangePolynomialBasis(0,start=0.0)
@test allequal(basis.points,[0.5])
@test allequal(basis(2.0),[1.0])
@test allequal(PB.derivative(basis,1.0),[0.0])

basis = PB.LagrangePolynomialBasis(1)
@test allequal(basis.points,[-1.0 1.0])
@test allequal(basis(-1.0),[1.0,0.0])
@test allequal(basis(1.0),[0.0,1.0])
@test allequal(PB.derivative(basis,-1.0),[-0.5,0.5])
@test allequal(PB.derivative(basis,1.0),[-0.5,0.5])
@test allequal(PB.gradient(basis,-1.0),[-0.5,0.5])
@test allequal(PB.gradient(basis,1.0),[-0.5,0.5])

basis = PB.LagrangePolynomialBasis(1,start=0.0)
@test allequal(basis.points,[0.0 1.0])
@test allequal(basis(0.0),[1.0,0.0])
@test allequal(basis(1.0),[0.0,1.0])
@test allequal(PB.derivative(basis,0.0),[-1.0,1.0])
@test allequal(PB.derivative(basis,1.0),[-1.0,1.0])
@test allequal(PB.gradient(basis,-1.0),[-1.0,1.0])
@test allequal(PB.gradient(basis,1.0),[-1.0,1.0])

basis = PB.LagrangePolynomialBasis(2)
@test allequal(basis.points,[-1.0 0.0 1.0])
@test allequal(basis(-1.0),[1.0,0.0,0.0])
@test allequal(basis(0.0),[0.0,1.0,0.0])
@test allequal(basis(1.0),[0.0,0.0,1.0])
@test allequal(PB.derivative(basis,-1.0),[-1.5,+2,-0.5])
@test allequal(PB.derivative(basis,0.0),[-0.5,0.0,0.5])
@test allequal(PB.derivative(basis,1.0),[0.5,-2,1.5])
@test allequal(PB.derivative(basis,-1.0),PB.gradient(basis,-1.0))
@test allequal(PB.derivative(basis,0.0),PB.gradient(basis,0.0))
@test allequal(PB.derivative(basis,1.0),PB.gradient(basis,1.0))

basis = PB.LagrangePolynomialBasis(2,stop=0.0)
@test allequal(basis.points,[-1.0 -0.5 0.0])
@test allequal(basis(-1.0),[1.0,0.0,0.0])
@test allequal(basis(-0.5),[0.0,1.0,0.0])
@test allequal(basis(0.0),[0.0,0.0,1.0])
@test allequal(PB.derivative(basis,-1.0),[-3.,4,-1])
@test allequal(PB.derivative(basis,-0.5),[-1.0,0,1])
@test allequal(PB.derivative(basis,0.0),[1.0,-4,3])

@test PB.number_of_basis_functions(basis) == 3
@test typeof(basis) == PB.LagrangePolynomialBasis{3}
@test supertype(PB.LagrangePolynomialBasis) == PB.AbstractBasis{1}
