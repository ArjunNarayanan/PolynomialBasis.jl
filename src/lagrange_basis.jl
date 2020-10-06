import Base: ==


abstract type AbstractBasis{dim,NF} end


struct LagrangePolynomialBasis{NF} <: AbstractBasis{1,NF}
    funcs::SP.PolynomialSystem{NF,1}
    points::SMatrix{1,NF}
    function LagrangePolynomialBasis(funcs::Vector{DP.Polynomial{C,R}},
        points::V) where {C,R<:Real,V<:AbstractVector}

        NF = length(funcs)
        order = NF - 1
        npoints = length(points)

        @assert NF > 0 "Vector of functions must be non-empty"
        @assert NF == npoints "Number of functions must be equal to number of interpolation points"
        @assert isapprox(sum(funcs),one(R)) "Polynomial basis must sum to unity"

        var = funcs[1].x.vars[1]
        for f in funcs
            @assert length(f.x.vars) == 1 "Require univariate polynomials"
            @assert var in f.x.vars "Require same variable in all polynomials"
            ord = maximum(maximum.(f.x.Z))
            @assert ord == order "Require all polynomials of same order"
        end
        polysystem = SP.PolynomialSystem(funcs)
        static_points = SMatrix{1,NF}(points')
        new{NF}(polysystem,static_points)
    end
end

function Base.show(io::IO,basis::LagrangePolynomialBasis{NF}) where {NF}
    p = order(basis)
    print(io, "1-D LagrangePolynomialBasis\n\tOrder: $p")
end

function order(basis::LagrangePolynomialBasis{NF}) where {NF}
    return NF-1
end

function LagrangePolynomialBasis(order::Z;start = -1.0, stop = 1.0) where {Z<:Integer}
    @assert start < stop
    @assert order >= 0

    DP.@polyvar x
    NF = order+1
    points = NF == 1 ? [0.5*(start+stop)] : range(start,stop=stop,length=NF)
    funcs = lagrange_polynomials(x,points)
    return LagrangePolynomialBasis(funcs,points)
end

function (B::LagrangePolynomialBasis)(x)
    return SP.evaluate(B.funcs, @SVector [x])
end

function number_of_basis_functions(basis::T) where {T<:AbstractBasis{dim,NF}} where {dim,NF}
    return NF
end

function derivative(B::LagrangePolynomialBasis{NF},x) where {NF}
    vals = SP.jacobian(B.funcs,@SVector [x])
    return SVector{NF}(vals)
end

function gradient(B::LagrangePolynomialBasis,x)
    return derivative(B,x)
end
