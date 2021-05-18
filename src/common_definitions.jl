abstract type AbstractBasis{dim,NF} end

function dimension(basis::B) where {B<:AbstractBasis{D}} where {D}
    return D
end

function number_of_basis_functions(
    basis::T,
) where {T<:AbstractBasis{dim,NF}} where {dim,NF}
    return NF
end

function gradient(
    basis::B,
    dir,
    x::V,
) where {B<:AbstractBasis{2,NF},V<:AbstractVector} where {NF}
    @assert length(x) == 2
    return gradient(basis,dir,x[1],x[2])
end

function gradient(
    basis::B,
    x::V,
) where {B<:AbstractBasis{2,NF},V<:AbstractVector} where {NF}
    @assert length(x) == 2
    return gradient(basis,x[1],x[2])
end
