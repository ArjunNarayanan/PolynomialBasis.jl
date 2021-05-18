using Test
# using Revise
using PolynomialBasis

PB = PolynomialBasis

function allequal(v1, v2)
    return all(v1 .≈ v2)
end


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
