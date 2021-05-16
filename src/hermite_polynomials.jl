function hermite_polynomial_coefficient_matrix(left,right)
    [1.0   left   left^2   left^3
     0.0   1.0   2.0*left  3.0*(left^2)
     1.0   right  right^2  right^3
     0.0   1.0   2.0*right 3.0*right^2]
end



function hermite_polynomial_coefficients(left,right)
    M = hermite_polynomial_coefficient_matrix(left,right)
    coeffs = M\I
end

function hermite_polynomials(x,left,right)
    coeffs = hermite_polynomial_coefficients(left,right)
    order,numbasis = size(coeffs)
    return [polynomial_from_coefficients(x,coeffs[:,i]) for i = 1:numbasis]
end
