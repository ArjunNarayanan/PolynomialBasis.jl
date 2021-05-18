using Test
import DynamicPolynomials: @polyvar
using Revise
using PolynomialBasis

PB = PolynomialBasis

@polyvar x

function allequal(v1,v2,tol)
    return allequal(v1,v2;tol=tol)
end

function allequal(v1,v2;tol=1e3eps())
    np = length(v1)
    f = length(v2) == np
    return f && all([isapprox(v1[i],v2[i],atol=tol) for i = 1:np])
end

funcs = [2.0+0.0x]
deleteat!(funcs,1)
@test_throws AssertionError PB.LagrangePolynomialBasis(funcs,zeros(0))

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
@test_throws MethodError PB.LagrangePolynomialBasis(2.5)
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
@test allequal(basis.points,[-1.0,1.0])
@test allequal(basis(-1.0),[1.0,0.0])
@test allequal(basis(1.0),[0.0,1.0])
@test allequal(PB.derivative(basis,-1.0),[-0.5,0.5])
@test allequal(PB.derivative(basis,1.0),[-0.5,0.5])
@test allequal(PB.gradient(basis,-1.0),[-0.5,0.5])
@test allequal(PB.gradient(basis,1.0),[-0.5,0.5])

basis = PB.LagrangePolynomialBasis(1,start=0.0)
@test allequal(basis.points,[0.0,1.0])
@test allequal(basis(0.0),[1.0,0.0])
@test allequal(basis(1.0),[0.0,1.0])
@test allequal(PB.derivative(basis,0.0),[-1.0,1.0])
@test allequal(PB.derivative(basis,1.0),[-1.0,1.0])
@test allequal(PB.gradient(basis,-1.0),[-1.0,1.0])
@test allequal(PB.gradient(basis,1.0),[-1.0,1.0])

basis = PB.LagrangePolynomialBasis(2)
@test allequal(basis.points,[-1.0,0.0,1.0])
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
@test allequal(basis.points,[-1.0,-0.5,0.0])
@test allequal(basis(-1.0),[1.0,0.0,0.0])
@test allequal(basis(-0.5),[0.0,1.0,0.0])
@test allequal(basis(0.0),[0.0,0.0,1.0])
@test allequal(PB.derivative(basis,-1.0),[-3.,4,-1])
@test allequal(PB.derivative(basis,-0.5),[-1.0,0,1])
@test allequal(PB.derivative(basis,0.0),[1.0,-4,3])

@test PB.number_of_basis_functions(basis) == 3
@test typeof(basis) == PB.LagrangePolynomialBasis{3,typeof(1.0)}
@test supertype(PB.LagrangePolynomialBasis) == PB.AbstractBasis{1}


basis = PB.LagrangePolynomialBasis(1)
@test allequal(PB.hessian(basis,-1),[0.,0.])
@test allequal(PB.hessian(basis,+1),[0.,0.])

basis = PB.LagrangePolynomialBasis(2)
@test allequal(PB.hessian(basis,-1),[1,-2,1])
@test allequal(PB.hessian(basis,0),[1,-2,1])
@test allequal(PB.hessian(basis,1),[1,-2,1])

basis = PB.LagrangePolynomialBasis(3)
coeffs = [x^3 for x in basis.points]
h = PB.hessian(basis,-1)
@test allequal(coeffs'*h,[-6.0])
h = PB.hessian(basis,0)
@test allequal(coeffs'*h,[0.0])
h = PB.hessian(basis,+1)
@test allequal(coeffs'*h,[+6.0])

basis = PB.LagrangePolynomialBasis(4)
coeffs = [x^4 for x in basis.points]
@test all([allequal(coeffs'*PB.hessian(basis,p),[12p^2],100eps()) for p in basis.points])
