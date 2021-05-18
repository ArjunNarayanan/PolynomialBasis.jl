struct InterpolatingPolynomial{N,B,T}
    coeffs::Matrix{T}
    basis::B
    function InterpolatingPolynomial(
        coeffs::Matrix{T},
        basis::B,
    ) where {B<:AbstractBasis{dim}} where {dim,T}

        N, NF = size(coeffs)
        @assert number_of_basis_functions(basis) == NF
        new{N,B,T}(coeffs, basis)

    end
end

function InterpolatingPolynomial(
    T::Type{<:Number},
    N::Z,
    basis::B,
) where {Z<:Integer} where {B<:AbstractBasis{dim,NF}} where {dim,NF}

    coeffs = zeros(T, N, NF)
    return InterpolatingPolynomial(coeffs, basis)

end

function InterpolatingPolynomial(
    N::Z,
    basis::B,
) where {Z<:Integer} where {B<:AbstractBasis}
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
    basis = LagrangeTensorProductBasis(dim, order, start = start, stop = stop)
    return InterpolatingPolynomial(T, N, basis)
end

function Base.show(io::IO, poly::InterpolatingPolynomial{N,B,T}) where {N,B,T}
    p = order(poly.basis)
    dim = dimension(poly.basis)
    NF = number_of_basis_functions(poly.basis)
    basistype = typeof(poly.basis)
    str = "InterpolatingPolynomial\n\tDimension  : $dim\n\t"*
          "Basis Type : $basistype\n\t"*
          "Order      : $p\n\t"*
          "Num. Funcs.: $NF\n\t"*
          "DOFs/Func. : $N"
    print(io, str)
end

function dimension(poly::InterpolatingPolynomial)
    return dimension(poly.basis)
end

function update!(P::InterpolatingPolynomial, coeffs)
    P.coeffs[:] = coeffs[:]
end

function (P::InterpolatingPolynomial{1,B,T})(x, y) where {B<:AbstractBasis{2},T}
    return ((P.coeffs)*(P.basis(x, y)))[1]
end

function (P::InterpolatingPolynomial{1,B,T})(
    x::V,
) where {B<:AbstractBasis{2},T} where {V<:AbstractVector}
    @assert length(x) == 2
    return ((P.coeffs)*(P.basis(x[1], x[2])))[1]
end

function (P::InterpolatingPolynomial{N,B,T})(
    x,
    y,
) where {B<:AbstractBasis{2},N,T}
    return ((P.coeffs) * (P.basis(x, y)))
end

function (P::InterpolatingPolynomial{N,B,T})(
    x::V,
) where {B<:AbstractBasis{2},N,T} where {V<:AbstractVector}
    @assert length(x) == 2
    return ((P.coeffs) * (P.basis(x[1], x[2])))
end

function gradient(
    P::InterpolatingPolynomial{1,B,T},
    x,
) where {B<:AbstractBasis{1},T}
    return ((P.coeffs) * (gradient(P.basis, x)))
end

function gradient(
    P::InterpolatingPolynomial{1,B,T},
    x,
    y,
) where {B<:AbstractBasis{2},T}
    return ((P.coeffs) * (gradient(P.basis, x, y)))
end

function gradient(
    P::InterpolatingPolynomial{1,B,T},
    x::V,
) where {B<:AbstractBasis{2},T} where {V<:AbstractVector}
    return ((P.coeffs) * (gradient(P.basis, x)))
end

function hessian(
    P::InterpolatingPolynomial{1,B,T},
    x,
) where {B<:AbstractBasis{1},T}
    return ((P.coeffs) * (hessian(P.basis, x)))
end

function hessian(
    P::InterpolatingPolynomial{1,B,T},
    x,
    y,
) where {B<:AbstractBasis{2},T}

    return P.coeffs * hessian(P.basis, x, y)
end

function hessian(
    P::InterpolatingPolynomial{1,B,T},
    x::V,
) where {B<:AbstractBasis{2},T} where {V<:AbstractVector}
    return ((P.coeffs) * (hessian(P.basis, x)))
end
