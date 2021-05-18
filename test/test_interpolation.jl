using Test
using Revise
using PolynomialBasis

PB = PolynomialBasis

function allequal(v1, v2)
    return all(v1 .≈ v2)
end

function allequal(v1, v2, tol)
    np = length(v1)
    f = length(v2) == np
    return f && all([isapprox(v1[i], v2[i], atol = tol) for i = 1:np])
end

coeffs = reshape([1.0], 1, 1)
b0 = PB.LagrangePolynomialBasis(0)
b1 = PB.LagrangePolynomialBasis(1)
tp1 = PB.LagrangeTensorProductBasis(2, 1)

@test_throws AssertionError PB.InterpolatingPolynomial(coeffs, b1)
ip = PB.InterpolatingPolynomial(coeffs, b0)

@test allequal(ip.coeffs, coeffs)

float_type = typeof(0.0)
ip = PB.InterpolatingPolynomial(float_type, 2, b1)
@test size(ip.coeffs) == (2, 2)

ip = PB.InterpolatingPolynomial(3, tp1)
@test size(ip.coeffs) == (3, 4)

ip = PB.InterpolatingPolynomial(1, 1, 2; start = 0.0)
@test allequal(ip.basis(0.0), [1.0, 0.0, 0.0])
@test allequal(ip.basis(0.5), [0.0, 1.0, 0.0])
@test allequal(ip.basis(1.0), [0.0, 0.0, 1.0])

v = [1.0 2.0 3.0]
PB.update!(ip, v)
@test allequal(ip.coeffs, v)
@test_throws DimensionMismatch PB.update!(ip, 1:4)

ip = PB.InterpolatingPolynomial(1, 2, 2)
PB.update!(ip, repeat([1.0, 2.0, 3.0], 3))
@test ip(0.0, -1.0) ≈ 1.0
@test ip(1.0, 0.0) ≈ 2.0
@test ip(-1.0, 1.0) ≈ 3.0
@test ip([-1.0, 1.0]...) ≈ 3.0

ip = PB.InterpolatingPolynomial(2, 2, 2)
coeffs = [
    1.0 1.0 1.0 2.0 2.0 2.0 3.0 3.0 3.0
    5.0 5.0 5.0 7.0 7.0 7.0 9.0 9.0 9.0
]
PB.update!(ip, coeffs)
@test allequal(ip(-1.0, 0.0), [1.0, 5.0])
@test allequal(ip(0.0, -1.0), [2.0, 7.0])
@test allequal(ip(1.0, 1.0), [3.0, 9.0])

P = PB.InterpolatingPolynomial(1, 1, 2)
f(x) = 3x^2 + 2x - 5
df(x) = 6x + 2
coeffs = f.(P.basis.points)
PB.update!(P, coeffs)
@test all(PB.gradient(P, [-1.0]) .≈ [df(-1.0)])
@test all(PB.gradient(P, [0.19]) .≈ [df(0.19)])

P = PB.InterpolatingPolynomial(1, 2, 2)
f(x, y) = 3x^2 * y + 2y^2 + 2x * y
dfx(x, y) = 6 * x * y + 2 * y
dfy(x, y) = 3x^2 + 4y + 2x
coeffs = [f((P.basis.points[:, i])...) for i = 1:9]
PB.update!(P, coeffs)
@test allequal(PB.gradient(P, [0.1, 0.2]), [dfx(0.1, 0.2) dfy(0.1, 0.2)])
@test allequal(PB.gradient(P, 1, [0.7, -0.3]), dfx(0.7, -0.3))
@test allequal(PB.gradient(P, 2, [-0.5, 0.75]), dfy(-0.5, 0.75))
@test allequal(PB.gradient(P, [0.1, 0.45]), [dfx(0.1, 0.45) dfy(0.1, 0.45)])


poly = PB.InterpolatingPolynomial(1, 1, 3)
f(x) = 7x^3 + 2x^2
coeffs = f.(poly.basis.points)
update!(poly, coeffs)
testh(x) = 42x + 4
@test all([
    allequal(PB.hessian(poly, p)[1], testh(p)) for p in poly.basis.points
])


poly = PB.InterpolatingPolynomial(1, 2, 2)
f(x) = 12x[1]^2 + 7x[1] * x[2] + 11x[2]^2
h = [24, 7, 22]
coeffs = mapslices(f, Array(poly.basis.points), dims = 1)
update!(poly, coeffs)
p = poly.basis.points
@test all([allequal(PB.hessian(poly, p[:, i]), h, 1e2eps()) for i = 1:9])


poly = PB.InterpolatingPolynomial(1, 2, 3)
f(x) = x[1]^3 + 2x[1]^2 * x[2] + 18.0
tp3h(x) = [6x[1] + 4x[2], 4x[1], 0.0]
p = poly.basis.points
coeffs = mapslices(f, Array(p), dims = 1)
update!(poly, coeffs)
@test all([
    allequal(PB.hessian(poly, p[:, i]), tp3h(p[:, i]), 1e3eps()) for i = 1:16
])


# Test hermite cubic interpolation
function f(v)
    x = v[1]
    y = v[2]
    return x^3 + 3y^3 + 2x^2*y + 8x * y
end

function fx(v)
    x = v[1]
    y = v[2]
    return 3x^2 + 4x*y + 8y
end

function fxx(v)
    x = v[1]
    y = v[2]
    return 6*x+4*y
end

function fy(v)
    x = v[1]
    y = v[2]
    return 9y^2 + 2x^2 + 8x
end

function fyy(v)
    x = v[1]
    y = v[2]
    return 18y
end

function fxy(v)
    x = v[1]
    y = v[2]
    return 4x+8
end

function testhessian(v)
    hxx = fxx(v)
    hxy = fxy(v)
    hyy = fyy(v)
    return [hxx,hxy,hyy]
end

function testgradient(v)
    gx = fx(v)
    gy = fy(v)
    return [gx,gy]
end

hermitetp = PB.HermiteTensorProductBasis(2)
interp = PB.InterpolatingPolynomial(1,hermitetp)
points = [-1.0  -1.0   1.0   1.0
          -1.0   1.0  -1.0   1.0]
v = vec(mapslices(f,points,dims=1))
vx = vec(mapslices(fx,points,dims=1))
vy = vec(mapslices(fy,points,dims=1))
vxy = vec(mapslices(fxy,points,dims=1))

coeffs = zeros(16)
coeffs[[1,5,9,13]] .= v
coeffs[[2,6,10,14]] .= vy
coeffs[[3,7,11,15]] .= vx
coeffs[[4,8,12,16]] .= vxy

update!(interp,coeffs)

testp = range(-1.0,stop=1.0,length=4)'
p = PB.tensor_product_points(2,testp)

vals = mapslices(interp,p,dims=1)
testvals = mapslices(f,p,dims=1)
@test allequal(vals,testvals)

grads = mapslices(x->vec(gradient(interp,x)),p,dims=1)
testgrads = mapslices(testgradient,p,dims=1)
@test allequal(grads,testgrads)

hess = mapslices(x->vec(hessian(interp,x)),p,dims=1)
testhess = mapslices(testhessian,p,dims=1)
@test allequal(hess,testhess)
