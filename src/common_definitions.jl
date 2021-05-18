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
    x::V,
) where {B<:AbstractBasis{dim,NF},V<:AbstractVector} where {dim,NF}
    @assert length(x) == dim
    
end
