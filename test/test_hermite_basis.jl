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

basis = PB.CubicHermiteBasis()
@test allequal(basis.left,-1.0)
@test allequal(basis.right,1.0)
@test allequal(basis(-1.0),[1.0,0.,0.,0.])
@test allequal(basis(1.0),[0.0,0.,1.,0.])
@test allequal(PB.derivative(basis,-1.0),[0.,1.,0.,0.])
@test allequal(PB.derivative(basis,1.0),[0.,0.,0.,1.])
@test allequal(PB.gradient(basis,-1.0),[0.,1.,0.,0.])
@test allequal(PB.gradient(basis,1.0),[0.,0.,0.,1.])
