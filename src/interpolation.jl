struct InterpolatingPolynomial{N,NF,B,T}
    coeffs::MMatrix{N,NF,T}
    basis::B
    function InterpolatingPolynomial(coeffs::MMatrix{N,NF,T},
        basis::B) where {B<:AbstractBasis{dim,NF}} where {N,NF,dim,T}

        new{N,NF,B,T}(coeffs,basis)

    end
end

function InterpolatingPolynomial(T::Type{<:Number},
    N::Z, basis::B) where {Z<:Integer} where {B<:AbstractBasis{dim,NF}} where {dim,NF}

    coeffs = MMatrix{N,NF}(zeros(T,N,NF))
    return InterpolatingPolynomial(coeffs,basis)

end

function InterpolatingPolynomial(N::Z,basis::B) where {Z<:Integer} where {B<:AbstractBasis}
    T = typeof(0.0)
    return InterpolatingPolynomial(T,N,basis)
end

function InterpolatingPolynomial(N::Z,dim::Z,order::Z;start = -1.0, stop = 1.0) where {Z<:Integer}
    T = typeof(0.0)
    basis = TensorProductBasis(dim,order,start=start,stop=stop)
    return InterpolatingPolynomial(T,N,basis)
end

function update!(P::InterpolatingPolynomial,coeffs)
    P.coeffs[:] = coeffs[:]
end

function (P::InterpolatingPolynomial{1})(x...)
    return ((P.coeffs)*(P.basis(x...)))[1]
end

function (P::InterpolatingPolynomial)(x...)
    return ((P.coeffs)*(P.basis(x...)))
end

function gradient(P::InterpolatingPolynomial{1},x::T) where {T<:Number}
    return ((P.coeffs)*(gradient(P.basis,x...)))[1]
end

function gradient(P::InterpolatingPolynomial{1},dir::Z,x...) where {Z<:Integer}
    return ((P.coeffs)*(gradient(P.basis,dir,x...)))[1]
end

function gradient(P::InterpolatingPolynomial,x...)
    return ((P.coeffs)*(gradient(P.basis,x...)))
end

function gradient(P::InterpolatingPolynomial,dir::Z,x...) where {Z<:Integer}
    return ((P.coeffs)*(gradient(P.basis,dir,x...)))
end
