using Test
using DynamicPolynomials
# using Revise
using PolynomialBasis

function allequal(v1,v2)
    return all(v1 .≈ v2)
end


coeffs = PolynomialBasis.hermite_polynomial_coefficients(-1.0,1.0)
testcoeffs = 0.25*[2    1  2  -1
                   -3  -1  3  -1
                   0   -1  0  +1
                   1    1 -1  +1]
@test allequal(coeffs,testcoeffs)

@polyvar x

hermitepolynomials = PolynomialBasis.hermite_polynomials(x,-1.0,1.0)
derivatives = differentiate.(hermitepolynomials,x)
v1 = [f(-1.0) for f in hermitepolynomials]
v2 = [f(-1.0) for f in derivatives]
v3 = [f(1.0) for f in hermitepolynomials]
v4 = [f(1.0) for f in derivatives]
@test allequal(v1,[1.0,0.,0.,0.])
@test allequal(v2,[0.,1.,0.,0.])
@test allequal(v3,[0.,0.,1.,0.])
@test allequal(v4,[0.,0.,0.,1.])


hermitepolynomials = PolynomialBasis.hermite_polynomials(x,0.0,1.0)
derivatives = differentiate.(hermitepolynomials,x)
v1 = [f(0.0) for f in hermitepolynomials]
v2 = [f(0.0) for f in derivatives]
v3 = [f(1.0) for f in hermitepolynomials]
v4 = [f(1.0) for f in derivatives]
@test allequal(v1,[1.0,0.,0.,0.])
@test allequal(v2,[0.,1.,0.,0.])
@test allequal(v3,[0.,0.,1.,0.])
@test allequal(v4,[0.,0.,0.,1.])
