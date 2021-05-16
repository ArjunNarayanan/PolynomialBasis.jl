struct CubicHermiteBasis{T} <: AbstractBasis{1,4,T}
    funcs::SP.PolynomialSystem{4,1}
    left::T
    right::T
    function CubicHermiteBasis(
        funcs::Vector{DP.Polynomial{C,R}},
        left::T,
        right::T,
    ) where {C,R,T}

        @assert length(funcs) == 4
        @assert left < right
        @assert isapprox(funcs[1] + funcs[3],one(R))

        polysystem = SP.PolynomialSystem(funcs)
        new{T}(polysystem,left,right)
    end
end

function Base.show(io::IO,basis::CubicHermiteBasis{T}) where {T}
    print(io,"1-D HermitePolynomialBasis")
end

function order(basis::CubicHermiteBasis{T}) where {T}
    return 3
end

function CubicHermiteBasis(;start = -1.0, stop = 1.0)
    @assert start < stop

    DP.@polyvar x
    funcs = hermite_polynomials(x,start,stop)
    return CubicHermiteBasis(funcs,start,stop)
end

function (B::CubicHermiteBasis{T})(x) where {T}
    return SP.evaluate(B.funcs,[x])
end

function derivative(B::CubicHermiteBasis{T},x) where {T}
    vals = SP.jacobian(B.funcs,[x])
    return vals
end

function gradient(B::CubicHermiteBasis{T},x) where {T}
    return derivative(B,x)
end

function hessian(B::CubicHermiteBasis{T},x) where {T}
    h = zeros(4,1,1)
    SP.hessian!(h,B.funcs,[x])
    return h[:,1,1]
end
