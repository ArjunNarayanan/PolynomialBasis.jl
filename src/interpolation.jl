struct InterpolatingPolynomial{N,NF,B,T}
    coeffs::MMatrix{N,NF,T}
    basis::B
    function InterpolatingPolynomial(
        coeffs::MMatrix{N,NF,T},
        basis::B,
    ) where {B<:AbstractBasis{dim,NF}} where {N,NF,dim,T}

        new{N,NF,B,T}(coeffs, basis)

    end
end

function InterpolatingPolynomial(
    T::Type{<:Number},
    N::Z,
    basis::B,
) where {Z<:Integer} where {B<:AbstractBasis{dim,NF}} where {dim,NF}

    coeffs = MMatrix{N,NF}(zeros(T, N, NF))
    return InterpolatingPolynomial(coeffs, basis)

end

function InterpolatingPolynomial(N::Z, basis::B) where {Z<:Integer} where {B<:AbstractBasis}
    T = typeof(0.0)
    return InterpolatingPolynomial(T, N, basis)
end

function InterpolatingPolynomial(
    N::Z,
    dim::Z,
    order::Z;
    start = -1.0,
    stop = 1.0,
) where {Z<:Integer}
    T = typeof(0.0)
    basis = TensorProductBasis(dim, order, start = start, stop = stop)
    return InterpolatingPolynomial(T, N, basis)
end

function Base.show(io::IO, poly::InterpolatingPolynomial{N,NF,B,T}) where {N,NF,B,T}
    p = order(poly.basis)
    dim = dimension(poly.basis)
    str = "InterpolatingPolynomial\n\tDimension  : $dim\n\tOrder      : $p\n\tNum. Funcs.: $NF\n\tDOFs/Func. : $N"
    print(io, str)
end

function dimension(poly::InterpolatingPolynomial)
    return dimension(poly.basis)
end

function update!(P::InterpolatingPolynomial, coeffs)
    P.coeffs[:] = coeffs[:]
end

function (P::InterpolatingPolynomial{1,NF,B,T})(x, y) where {B<:AbstractBasis{2},NF,T}
    return ((P.coeffs)*(P.basis(x, y)))[1]
end

function (P::InterpolatingPolynomial{1,NF,B,T})(
    x::V,
) where {B<:AbstractBasis{2},NF,T} where {V<:AbstractVector}
    @assert length(x) == 2
    return ((P.coeffs)*(P.basis(x[1], x[2])))[1]
end

function (P::InterpolatingPolynomial{N,NF,B,T})(x, y) where {B<:AbstractBasis{2},N,NF,T}
    return ((P.coeffs) * (P.basis(x, y)))
end

function (P::InterpolatingPolynomial{N,NF,B,T})(
    x::V,
) where {B<:AbstractBasis{2},N,NF,T} where {V<:AbstractVector}
    @assert length(x) == 2
    return ((P.coeffs) * (P.basis(x[1], x[2])))
end

function gradient(P::InterpolatingPolynomial{1,NF,B,T}, x) where {B<:AbstractBasis{1},NF,T}
    return ((P.coeffs) * (gradient(P.basis, x)))
end

function gradient(
    P::InterpolatingPolynomial{1,NF,B,T},
    x,
    y,
) where {B<:AbstractBasis{2},NF,T}
    return ((P.coeffs) * (gradient(P.basis, x, y)))
end

function gradient(
    P::InterpolatingPolynomial{1,NF,B,T},
    x::V,
) where {B<:AbstractBasis{2},NF,T} where {V<:AbstractVector}
    return ((P.coeffs) * (gradient(P.basis, x)))
end

function hessian(P::InterpolatingPolynomial{1,NF,B,T}, x) where {B<:AbstractBasis{1},NF,T}
    return ((P.coeffs) * (hessian(P.basis, x)))
end

function hessian(
    P::InterpolatingPolynomial{1,NF,B,T},
    x,
    y,
) where {B<:AbstractBasis{2},NF,T}

    return P.coeffs * hessian(P.basis, x, y)
end

function hessian(
    P::InterpolatingPolynomial{1,NF,B,T},
    x::V,
) where {B<:AbstractBasis{2},NF,T} where {V<:AbstractVector}
    return ((P.coeffs) * (hessian(P.basis, x)))
end
