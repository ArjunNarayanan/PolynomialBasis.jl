import Base: ==


struct LagrangeBasis{NF,T} <: AbstractBasis{1,NF}
    funcs::SP.PolynomialSystem{NF,1}
    points::Vector{T}
    function LagrangeBasis(funcs::Vector{DP.Polynomial{C,R}},
        points::Vector{T}) where {C,R,T}

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
        new{NF,T}(polysystem,points)
    end
end

function type_of_interpolation_points(B::LagrangeBasis{NF,T}) where {NF,T}
    return T
end

function Base.show(io::IO,basis::LagrangeBasis{NF,T}) where {NF,T}
    p = order(basis)
    print(io, "1-D LagrangeBasis\n\tOrder: $p")
end

function order(basis::LagrangeBasis{NF,T}) where {NF,T}
    return NF-1
end

function LagrangeBasis(order::Z;start = -1.0, stop = 1.0) where {Z<:Integer}
    @assert start < stop
    @assert order >= 0

    DP.@polyvar x
    NF = order+1
    points = NF == 1 ? [0.5*(start+stop)] : Array(range(start,stop=stop,length=NF))
    funcs = lagrange_polynomials(x,points)
    return LagrangeBasis(funcs,points)
end

function (B::LagrangeBasis)(x)
    return SP.evaluate(B.funcs,[x])
end

function derivative(B::LagrangeBasis{NF,T},x) where {NF,T}
    vals = SP.jacobian(B.funcs,[x])
    return vals
end

function gradient(B::LagrangeBasis{NF,T},x) where {NF,T}
    return derivative(B,x)
end

function hessian(B::LagrangeBasis{NF,T},x) where {NF,T}
    h = zeros(NF,1,1)
    SP.hessian!(h,B.funcs,[x])
    return h[:,1,1]
end
