abstract type AbstractTensorProductBasis{dim,T,NF} <: AbstractBasis{dim,NF} end

struct TensorProductBasis{dim,T,NF} <: AbstractTensorProductBasis{dim,T,NF}
    basis::T
    points::SMatrix{dim,NF}
    function TensorProductBasis(dim::Z,B::T) where {Z<:Integer} where {T<:AbstractBasis{1}}
        @assert 1 <= dim <= 3
        N1D = number_of_basis_functions(B)
        NF = N1D^dim
        points = tensor_product_points(dim,B.points)
        new{dim,T,NF}(B,points)
    end
end

function TensorProductBasis(dim,order; start = -1.0, stop = 1.0)
    basis_1d = LagrangePolynomialBasis(order, start = start, stop = stop)
    return TensorProductBasis(dim,basis_1d)
end

function (B::TensorProductBasis{1})(x::T) where {T<:Number}
    return B.basis(x)
end

function (B::TensorProductBasis{1})(x::V) where {V<:AbstractVector}
    @assert length(x) == 1
    return B(x[1])
end

function (B::TensorProductBasis{2})(x,y)
    return kron(B.basis(x),B.basis(y))
end

function (B::TensorProductBasis{2})(x::V) where {V<:AbstractVector}
    @assert length(x) == 2
    return B(x[1],x[2])
end

function (B::TensorProductBasis{3})(x,y,z)
    return kron(B.basis(x),B.basis(y),B.basis(z))
end

function (B::TensorProductBasis{3})(x::V) where {V<:AbstractVector}
    @assert length(x) == 3
    return B(x[1],x[2],x[3])
end

function gradient(B::TensorProductBasis{1},x)
    return derivative(B.basis,x)
end

function gradient(B::TensorProductBasis{1},x::V) where {V<:AbstractVector}
    @assert length(x) == 1
    return gradient(B,x[1])
end

function gradient(B::TensorProductBasis{2},dir::Z,x,y) where {Z<:Integer}
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

function gradient(B::TensorProductBasis{2},x,y)
    col1 = gradient(B,1,x,y)
    col2 = gradient(B,2,x,y)
    return hcat(col1,col2)
end

function gradient(B::TensorProductBasis{3},dir::Z,x,y,z) where {Z<:Integer}
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

function gradient(B::TensorProductBasis{3},x,y,z)
    Nx = B.basis(x)
    Ny = B.basis(y)
    Nz = B.basis(z)

    dNx = derivative(B.basis,x)
    dNy = derivative(B.basis,y)
    dNz = derivative(B.basis,z)

    return hcat(kron(dNx,Ny,Nz),kron(Nx,dNy,Nz),kron(Nx,Ny,dNz))
end

function gradient(B::TensorProductBasis{2},dir::Z,x::V) where {Z<:Integer} where {V<:AbstractVector}
    @assert length(x) == 2
    @assert 1 <= dir <= 2
    return gradient(B,dir,x[1],x[2])
end

function gradient(B::TensorProductBasis{2},x::V) where {V<:AbstractVector}
    @assert length(x) == 2
    return gradient(B,x[1],x[2])
end

function gradient(B::TensorProductBasis{3},dir::Z,x::V) where {Z<:Integer} where {V<:AbstractVector}
    @assert length(x) == 3
    @assert 1 <= dir <= 3
    return gradient(B,dir,x[1],x[2],x[3])
end

function gradient(B::TensorProductBasis{3},x::V) where {V<:AbstractVector}
    @assert length(x) == 3
    return gradient(B,x[1],x[2],x[3])
end

function number_of_1d_points(p::V) where {V<:AbstractVector}
    @assert length(p) == 1
    return 1
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
