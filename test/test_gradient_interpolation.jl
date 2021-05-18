using Test
using Revise
using PolynomialBasis

function interpolating_gradient(poly)
    order = PolynomialBasis.order(poly)
    interpgrad = InterpolatingPolynomial(2, 2, order)

    points =
        PolynomialBasis.interpolation_points(PolynomialBasis.basis(interpgrad))
    interpvals = mapslices(x -> vec(gradient(poly, x)), points, dims = 1)

    PolynomialBasis.update!(interpgrad, interpvals)
    return interpgrad
end

function allapprox(u, v)
    return all(u .≈ v)
end

function linear(coeffs, v)
    x, y = v
    a, b, c, d = coeffs

    return a * x * y + b * x + c * y + d
end

function gradlinear(coeffs, v)
    x, y = v
    a, b, c, d = coeffs

    fx = a * y + b
    fy = a * x + c
    return [fx, fy]
end

coeffs = [1.0, 6.0, 2.0, 7.0]
poly = InterpolatingPolynomial(1, 2, 1)
points = PolynomialBasis.interpolation_points(PolynomialBasis.basis(poly))
interpcoeffs = vec(mapslices(x -> linear(coeffs, x), points, dims = 1))
update!(poly, interpcoeffs)

testp = rand(2, 10)
vals = mapslices(poly, testp, dims = 1)
testvals = mapslices(x -> linear(coeffs, x), testp, dims = 1)
@test allapprox(vals, testvals)

interpgrad = interpolating_gradient(poly)
gvals = mapslices(interpgrad, testp, dims = 1)
testgvals = mapslices(x -> gradlinear(coeffs, x), testp, dims = 1)
@test allapprox(gvals, testgvals)


function quadratic(coeffs, v)
    x, y = v
    a, b, c, d, e, f, g, h, i = coeffs

    return a * x^2 * y^2 +
           b * x^2 * y +
           c * x^2 +
           d * x * y^2 +
           e * x * y +
           f * x +
           g * y^2 +
           h * y +
           i
end

function gradquadratic(coeffs, v)
    x, y = v
    a, b, c, d, e, f, g, h, i = coeffs

    fx = 2 * a * x * y^2 + 2 * b * x * y + 2 * c * x + d * y^2 + e * y + f
    fy = 2 * a * x^2 * y + b * x^2 + 2 * d * x * y + e * x + 2 * g * y + h

    return [fx, fy]
end

coeffs = [3.0, 4.0, 1.0, 11.0, 7.0, 3.0, 5.0, 13.0, 15.0]
poly = InterpolatingPolynomial(1, 2, 2)
points = PolynomialBasis.interpolation_points(PolynomialBasis.basis(poly))
interpcoeffs = vec(mapslices(x -> quadratic(coeffs, x), points, dims = 1))
update!(poly, interpcoeffs)

testp = rand(2, 10)
vals = mapslices(poly, testp, dims = 1)
testvals = mapslices(x -> quadratic(coeffs, x), testp, dims = 1)
@test allapprox(vals, testvals)

interpgrad = interpolating_gradient(poly)
gvals = mapslices(interpgrad, testp, dims = 1)
testgvals = mapslices(x -> gradquadratic(coeffs, x), testp, dims = 1)
@test allapprox(gvals, testgvals)

function cubic(coeffs, v)
    x, y = v
    a, b, c, d, e, f, g, h, i, j = coeffs

    return a * x^3 +
           b * x^2 * y +
           c * x * y^2 +
           d * y^3 +
           e * x^2 +
           f * x * y +
           g * y^2 +
           h * x +
           i * y +
           j
end

function gradcubic(coeffs, v)
    x, y = v
    a, b, c, d, e, f, g, h, i, j = coeffs

    fx = 3 * a * x^2 + 2 * b * x * y + c * y^2 + 2 * e * x + f * y + h
    fy = b * x^2 + 2 * c * x * y + 3 * d * y^2 + f * x + 2 * g * y + i

    return [fx,fy]
end

# coeffs = [7.,2,4,9,6,3,5,7,1,11]
coeffs = [10,12,6,17,9,5,2,1,8,11.0]
poly = InterpolatingPolynomial(1, 2, 3)
points = PolynomialBasis.interpolation_points(PolynomialBasis.basis(poly))
interpcoeffs = vec(mapslices(x -> cubic(coeffs, x), points, dims = 1))
update!(poly, interpcoeffs)

testp = rand(2, 10)
vals = mapslices(poly, testp, dims = 1)
testvals = mapslices(x -> cubic(coeffs, x), testp, dims = 1)
@test allapprox(vals, testvals)

interpgrad = interpolating_gradient(poly)
gvals = mapslices(interpgrad, testp, dims = 1)
testgvals = mapslices(x -> gradcubic(coeffs, x), testp, dims = 1)
@test allapprox(gvals, testgvals)
