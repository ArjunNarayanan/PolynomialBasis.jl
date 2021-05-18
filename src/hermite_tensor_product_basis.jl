struct HermiteTensorProductBasis{dim,NF,T} <: AbstractBasis{dim,NF}
    basis::CubicHermiteBasis{T}
    points::Matrix{T}
    function HermiteTensorProductBasis(dim; start = -1.0, stop = 1.0)
        basis1d = CubicHermiteBasis(start = start, stop = stop)
        NF = (number_of_basis_functions(basis1d))^dim
        points = hermite_tensor_product_interpolation_points(dim, start, stop)
        T = type_of_interpolation_points(basis1d)
        new{dim,NF,T}(basis1d, points)
    end
end

function Base.show(io::IO, basis::HermiteTensorProductBasis{dim,NF,T}) where {dim,NF,T}
    str = "HermiteTensorProductBasis\n\tDimension : $dim"
    print(io,str)
end

function hermite_tensor_product_interpolation_points(dim, start, stop)
    if dim == 2
        points = tensor_product_points(2, [start, stop]')
        return repeat(points, inner = (1, 4))
    else
        error("Current support for dim = $dim, got dim = $dim")
    end
end

function hermite_outer_product(u,v)
    return [
        u[1] * v[1],
        u[1] * v[2],
        u[2] * v[1],
        u[2] * v[2],
        u[1] * v[3],
        u[1] * v[4],
        u[2] * v[3],
        u[2] * v[4],
        u[3] * v[1],
        u[3] * v[2],
        u[4] * v[1],
        u[4] * v[2],
        u[3] * v[3],
        u[3] * v[4],
        u[4] * v[3],
        u[4] * v[4],
    ]
end

function (B::HermiteTensorProductBasis{2})(x, y)
    u = B.basis(x)
    v = B.basis(y)
    return hermite_outer_product(u,v)
end

function (B::HermiteTensorProductBasis{2})(x::V) where {V<:AbstractVector}
    @assert length(x) == 2
    return B([x[1],x[2]])
end

function gradient(B::HermiteTensorProductBasis{2},dir::Z,x,y) where {Z<:Integer}
    @assert 1 <= dir <= 2
    if dir == 1
        dNx = derivative(B.basis,x)
        Ny = B.basis(y)
        return hermite_outer_product(dNx,Ny)
    else
        Nx = B.basis(x)
        dNy = derivative(B.basis,y)
        return hermite_outer_product(Nx,dNy)
    end
end

function gradient(B::HermiteTensorProductBasis{2},x,y)
    col1 = gradient(B,1,x,y)
    col2 = gradient(B,2,x,y)
    return hcat(col1,col2)
end

function hessian(B::HermiteTensorProductBasis{2},x,y)

    d2Nx = hessian(B.basis,x)
    d1Nx = derivative(B.basis,x)
    d0Nx = B.basis(x)

    d2Ny = hessian(B.basis,y)
    d1Ny = derivative(B.basis,y)
    d0Ny = B.basis(y)

    Nxx = hermite_outer_product(d2Nx,d0Ny)
    Nxy = hermite_outer_product(d1Nx,d1Ny)
    Nyy = hermite_outer_product(d0Nx,d2Ny)

    return hcat(Nxx,Nxy,Nyy)
end

function hessian(B::HermiteTensorProductBasis{2},x::V) where {V<:AbstractVector}
    @assert length(x) == 2
    return hessian(B,x[1],x[2])
end
