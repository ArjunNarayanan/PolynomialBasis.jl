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


function LagrangePolynomialBasis(order::Z;start::R = -1.0, stop::R = 1.0) where {Z<:Integer,R<:Real}
    @assert start < stop
    @assert order >= 0

    DP.@polyvar x
    NF = order+1
    points = NF == 1 ? [0.0] : range(start,stop=stop,length=NF)
    funcs = lagrange_polynomials(x,points)
    return LagrangePolynomialBasis(funcs,points)
end

function (B::LagrangePolynomialBasis)(x)
    return SP.evaluate(B.funcs, @SVector [x])
end

function derivative(B::LagrangePolynomialBasis{NF},x) where {NF}
    vals = SP.jacobian(B.funcs,@SVector [x])
    return SVector{NF}(vals)
end
