function vandermonde_matrix(points::V) where {V<:AbstractVector}
    N = length(points)
    m = hcat([points.^i for i = 0:N-1]...)
    return m
end

function lagrange_polynomial_coefficients(points::V) where {V<:AbstractVector}
    matrix = vandermonde_matrix(points)
    return inv(matrix)
end

function polynomial_from_coefficients(x::DP.PolyVar,coeffs::V) where {V<:AbstractVector}

    N = length(coeffs)
    return sum([coeffs[i+1]*x^i for i = 0:N-1])

end

function lagrange_polynomials(x::DP.PolyVar,interpolation_points::V) where {V<:AbstractVector}

    N = length(interpolation_points)
    coeffs = lagrange_polynomial_coefficients(interpolation_points)
    return [polynomial_from_coefficients(x,coeffs[:,i]) for i = 1:N]

end
