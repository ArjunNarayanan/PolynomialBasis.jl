struct LagrangeTensorProductBasis{dim,NF,NF1D,T} <: AbstractBasis{dim,NF}
    basis::LagrangePolynomialBasis{NF1D,T}
    points::Matrix{T}
    function LagrangeTensorProductBasis(dim,order;start = -1.0,stop = 1.0)
        @assert 1 <= dim <= 3
        basis1d = LagrangePolynomialBasis(order, start = start, stop = stop)
        NF1D = number_of_basis_functions(basis1d)
        NF = NF1D^dim
        T = type_of_interpolation_points(basis1d)
        points = tensor_product_points(dim,basis1d.points')
        new{dim,NF,NF1D,T}(basis1d,points)
    end
end

function Base.show(io::IO, basis::LagrangeTensorProductBasis{dim,NF,NF1D,T}) where {dim,NF,NF1D,T}
    p = order(basis)
    str = "LagrangeTensorProductBasis\n\tDimension  : $dim\n\tOrder      : $p\n\tNum. Funcs.: $NF"
    print(io,str)
end

function order(basis::LagrangeTensorProductBasis)
    return order(basis.basis)
end

function (B::LagrangeTensorProductBasis{1})(x)
    return B.basis(x)
end

function (B::LagrangeTensorProductBasis{1})(x::V) where {V<:AbstractVector}
    @assert length(x) == 1
    return B(x[1])
end

function (B::LagrangeTensorProductBasis{2})(x,y)
    return kron(B.basis(x),B.basis(y))
end

function (B::LagrangeTensorProductBasis{2})(x::V) where {V<:AbstractVector}
    @assert length(x) == 2
    return B(x[1],x[2])
end

function (B::LagrangeTensorProductBasis{3})(x,y,z)
    return kron(B.basis(x),B.basis(y),B.basis(z))
end

function (B::LagrangeTensorProductBasis{3})(x::V) where {V<:AbstractVector}
    @assert length(x) == 3
    return B(x[1],x[2],x[3])
end

function gradient(B::LagrangeTensorProductBasis{1},x)
    return derivative(B.basis,x)
end

function gradient(B::LagrangeTensorProductBasis{1},x::V) where {V<:AbstractVector}
    @assert length(x) == 1
    return gradient(B,x[1])
end

function gradient(B::LagrangeTensorProductBasis{2},dir::Z,x,y) where {Z<:Integer}
    @assert 1 <= dir <= 2
    if dir == 1
        dNx = derivative(B.basis,x)
        Ny = B.basis(y)
        return kron(dNx,Ny)
    else
        Nx = B.basis(x)
        dNy = derivative(B.basis,y)
        return kron(Nx,dNy)
    end
end

function gradient(B::LagrangeTensorProductBasis{2},x,y)
    col1 = gradient(B,1,x,y)
    col2 = gradient(B,2,x,y)
    return hcat(col1,col2)
end

function gradient(B::LagrangeTensorProductBasis{2},dir,x::V) where {V<:AbstractVector}
    @assert length(x) == 2
    @assert 1 <= dir <= 2
    return gradient(B,dir,x[1],x[2])
end

function gradient(B::LagrangeTensorProductBasis{2},x::V) where {V<:AbstractVector}
    @assert length(x) == 2
    return gradient(B,x[1],x[2])
end

function gradient(B::LagrangeTensorProductBasis{3},dir,x,y,z)
    @assert 1 <= dir <= 3
    if dir == 1
        dNx = derivative(B.basis,x)
        Ny = B.basis(y)
        Nz = B.basis(z)
        return kron(dNx,Ny,Nz)
    elseif dir == 2
        Nx = B.basis(x)
        dNy = derivative(B.basis,y)
        Nz = B.basis(z)
        return kron(Nx,dNy,Nz)
    else
        Nx = B.basis(x)
        Ny = B.basis(y)
        dNz = derivative(B.basis,z)
        return kron(Nx,Ny,dNz)
    end
end

function gradient(B::LagrangeTensorProductBasis{3},x,y,z)
    Nx = B.basis(x)
    Ny = B.basis(y)
    Nz = B.basis(z)

    dNx = derivative(B.basis,x)
    dNy = derivative(B.basis,y)
    dNz = derivative(B.basis,z)

    return hcat(kron(dNx,Ny,Nz),kron(Nx,dNy,Nz),kron(Nx,Ny,dNz))
end

function gradient(B::LagrangeTensorProductBasis{3},dir,x::V) where {V<:AbstractVector}
    @assert length(x) == 3
    @assert 1 <= dir <= 3
    return gradient(B,dir,x[1],x[2],x[3])
end

function gradient(B::LagrangeTensorProductBasis{3},x::V) where {V<:AbstractVector}
    @assert length(x) == 3
    return gradient(B,x[1],x[2],x[3])
end

function hessian(B::LagrangeTensorProductBasis{1},x)
    return hessian(B.basis,x)
end

function hessian(B::LagrangeTensorProductBasis{1},x::V) where {V<:AbstractVector}
    @assert length(x) == 1
    return hessian(B,x[1])
end

function hessian(B::LagrangeTensorProductBasis{2},x,y)

    d2Nx = hessian(B.basis,x)
    d1Nx = derivative(B.basis,x)
    d0Nx = B.basis(x)

    d2Ny = hessian(B.basis,y)
    d1Ny = derivative(B.basis,y)
    d0Ny = B.basis(y)

    Nxx = kron(d2Nx,d0Ny)
    Nxy = kron(d1Nx,d1Ny)
    Nyy = kron(d0Nx,d2Ny)

    return hcat(Nxx,Nxy,Nyy)
end

function hessian(B::LagrangeTensorProductBasis{2},x::V) where {V<:AbstractVector}
    @assert length(x) == 2
    return hessian(B,x[1],x[2])
end

function number_of_1d_points(p::M) where {M<:AbstractMatrix}
    m,n = size(p)
    @assert m == 1
    return n
end

function tensor_product_points(N,points)
    @assert 1 <= N <= 3

    np = number_of_1d_points(points)

    if N == 1
        return points
    elseif N == 2
        row1 = repeat(points,inner=(1,np))
        row2 = repeat(points,outer=(1,np))
        return vcat(row1,row2)
    else
        np2 = np^2
        row1 = repeat(points,inner=(1,np2))
        row2 = repeat(repeat(points,inner=(1,np)),outer=(1,np))
        row3 = repeat(points,outer=(1,np2))
        return vcat(row1,row2,row3)
    end
end
