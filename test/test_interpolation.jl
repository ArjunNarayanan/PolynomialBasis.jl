using Test
using StaticArrays
# using Revise
using PolynomialBasis

PB = PolynomialBasis

function allequal(v1,v2)
    return all(v1 .≈ v2)
end

function allequal(v1,v2,tol)
    np = length(v1)
    f = length(v2) == np
    return f && all([isapprox(v1[i],v2[i],atol=tol) for i = 1:np])
end

coeffs = MMatrix{1,1}([1.0])
b0 = PB.LagrangePolynomialBasis(0)
b1 = PB.LagrangePolynomialBasis(1)
tp1 = PB.TensorProductBasis(2,1)

@test_throws MethodError PB.InterpolatingPolynomial(coeffs,b1)
ip = PB.InterpolatingPolynomial(coeffs,b0)

@test typeof(ip) == PB.InterpolatingPolynomial{1,1,PB.LagrangePolynomialBasis{1},Float64}
@test allequal(ip.coeffs,coeffs)
@test typeof(ip.basis) == typeof(b0)

float_type = typeof(0.0)
ip = PB.InterpolatingPolynomial(float_type,2,b1)
@test typeof(ip) == PB.InterpolatingPolynomial{2,2,PB.LagrangePolynomialBasis{2},float_type}
@test typeof(ip.coeffs) == MMatrix{2,2,float_type,4}
@test size(ip.coeffs) == (2,2)
@test typeof(ip.basis) == typeof(b1)

ip = PB.InterpolatingPolynomial(3,tp1)
@test size(ip.coeffs) == (3,4)
@test typeof(ip) == PB.InterpolatingPolynomial{3,4,PB.TensorProductBasis{2,PB.LagrangePolynomialBasis{2},4},float_type}

ip = PB.InterpolatingPolynomial(1,1,2;start=0.0)
@test allequal(ip.basis(0.0),[1.0,0.0,0.0])
@test allequal(ip.basis(0.5),[0.0,1.0,0.0])
@test allequal(ip.basis(1.0),[0.0,0.0,1.0])
@test typeof(ip) == PB.InterpolatingPolynomial{1,3,PB.TensorProductBasis{1,PB.LagrangePolynomialBasis{3},3},float_type}

v = [1.0 2.0 3.0]
PB.update!(ip,v)
@test allequal(ip.coeffs,v)
@test_throws DimensionMismatch PB.update!(ip,1:4)

ip = PB.InterpolatingPolynomial(1,2,2)
PB.update!(ip,repeat([1.0,2.0,3.0],3))
@test ip(0.0,-1.0) ≈ 1.0
@test ip(1.0,0.0) ≈ 2.0
@test ip(-1.0,1.0) ≈ 3.0
@test ip([-1.0,1.0]...) ≈ 3.0

ip = PB.InterpolatingPolynomial(2,2,2)
coeffs = [1.0 1.0 1.0 2.0 2.0 2.0 3.0 3.0 3.0
          5.0 5.0 5.0 7.0 7.0 7.0 9.0 9.0 9.0]
PB.update!(ip, coeffs)
@test allequal(ip(-1.0,0.0), [1.0,5.0])
@test allequal(ip(0.0,-1.0), [2.0,7.0])
@test allequal(ip(1.0,1.0), [3.0,9.0])

P = PB.InterpolatingPolynomial(1,1,2)
f(x) = 3x^2 + 2x - 5
df(x) = 6x + 2
coeffs = f.(P.basis.points)
PB.update!(P, coeffs)
@test all(PB.gradient(P, [-1.0]) .≈ [df(-1.0)])
@test all(PB.gradient(P, [0.19]) .≈ [df(0.19)])

P = PB.InterpolatingPolynomial(1,2,2)
f(x,y) = 3x^2*y + 2y^2 + 2x*y
dfx(x,y) = 6*x*y + 2*y
dfy(x,y) = 3x^2 + 4y + 2x
coeffs = [f((P.basis.points[:,i])...) for i = 1:9]
PB.update!(P, coeffs)
@test allequal(PB.gradient(P, [0.1, 0.2]), [dfx(0.1, 0.2) dfy(0.1,0.2)])
@test allequal(PB.gradient(P, 1, [0.7, -0.3]), dfx(0.7, -0.3))
@test allequal(PB.gradient(P, 2, [-0.5, 0.75]), dfy(-0.5, 0.75))
@test allequal(PB.gradient(P, [0.1, 0.45]), [dfx(0.1, 0.45) dfy(0.1, 0.45)])


poly = PB.InterpolatingPolynomial(1,1,3)
f(x) = 7x^3 + 2x^2
coeffs = f.(poly.basis.points)
update!(poly,coeffs)
testh(x) = 42x + 4
@test all([allequal(PB.hessian(poly,p)[1],testh(p)) for p in poly.basis.points])


poly = PB.InterpolatingPolynomial(1,2,2)
f(x) = 12x[1]^2 + 7x[1]*x[2] + 11x[2]^2
h = [24,7,22]
coeffs = mapslices(f,Array(poly.basis.points),dims=1)
update!(poly,coeffs)
p = poly.basis.points
@test all([allequal(PB.hessian(poly,p[:,i]),h,1e2eps()) for i = 1:9])


poly = PB.InterpolatingPolynomial(1,2,3)
f(x) = x[1]^3 + 2x[1]^2*x[2] + 18.0
tp3h(x) = [6x[1]+4x[2],4x[1],0.0]
p = poly.basis.points
coeffs = mapslices(f,Array(p),dims=1)
update!(poly,coeffs)
@test all([allequal(PB.hessian(poly,p[:,i]),tp3h(p[:,i]),1e3eps()) for i = 1:16])
