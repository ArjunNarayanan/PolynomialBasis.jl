using Test
# using Revise
using PolynomialBasis

PB = PolynomialBasis

function allequal(v1,v2)
    return all(v1 .≈ v2)
end

function allequal(v1,v2,tol)
    np = length(v1)
    f = length(v2) == np
    return f && all([isapprox(v1[i],v2[i],atol=tol) for i = 1:np])
end


p = [-1.0 1.0]
@test_throws AssertionError PB.tensor_product_points(0,p)
@test_throws AssertionError PB.tensor_product_points(4,p)

tp = PB.tensor_product_points(1,p)
@test allequal(p,tp)

tp = PB.tensor_product_points(2,p)
testtp = [-1.0 -1.0  1.0  1.0
          -1.0  1.0 -1.0  1.0]
@test allequal(tp,testtp)

tp = PB.tensor_product_points(3,p)
testtp = [-1.0   -1.0   -1.0   -1.0    1.0    1.0    1.0    1.0
          -1.0   -1.0    1.0    1.0   -1.0   -1.0    1.0    1.0
          -1.0    1.0   -1.0    1.0   -1.0    1.0   -1.0    1.0]
@test allequal(tp,testtp)

p = [1.0]
tp = PB.tensor_product_points(1,p')
@test allequal(p,tp)

tp = PB.tensor_product_points(2,p')
testtp = [1.0,1.0]
@test allequal(tp,testtp)

tp = PB.tensor_product_points(3,p')
testtp = [1.0;1.0;1.0]
@test allequal(tp,testtp)

basis = PB.LagrangePolynomialBasis(0)
@test_throws AssertionError PB.TensorProductBasis(0,basis)
@test_throws AssertionError PB.TensorProductBasis(4,basis)
tpb = PB.TensorProductBasis(1,basis)
@test typeof(tpb) == PB.TensorProductBasis{1,PB.LagrangePolynomialBasis{1,typeof(1.0)},1,typeof(1.0)}
@test allequal(tpb(0.0),[1.0])
@test allequal(PB.gradient(tpb,0.0),[0.0])
@test allequal(PB.gradient(tpb,[0.0]),[0.0])
@test_throws AssertionError PB.gradient(tpb,[0.0,0.0])

tpb = PB.TensorProductBasis(1,0)
@test typeof(tpb) == PB.TensorProductBasis{1,PB.LagrangePolynomialBasis{1,typeof(1.0)},1,typeof(1.0)}
@test allequal(tpb(0.0),[1.0])
@test allequal(PB.gradient(tpb,0.0),[0.0])
@test allequal(PB.gradient(tpb,[0.0]),[0.0])
@test_throws AssertionError PB.gradient(tpb,[0.0,0.0])

tpb = PB.TensorProductBasis(1,1)
@test typeof(tpb) == PB.TensorProductBasis{1,PB.LagrangePolynomialBasis{2,typeof(1.0)},2,typeof(1.0)}
@test allequal(tpb(-1.0),[1.0,0.0])
@test allequal(tpb(1.0),[0.0,1.0])
@test_throws AssertionError tpb([1.0,2.0])
@test allequal(tpb([1.0]),[0.0,1.0])
@test allequal(PB.gradient(tpb,-1.0),[-0.5,0.5])
@test allequal(PB.gradient(tpb,1.0),[-0.5,0.5])
@test_throws AssertionError PB.gradient(tpb,[1.0,2.0])
@test allequal(PB.gradient(tpb,[1.0]),[-0.5,0.5])

tpb = PB.TensorProductBasis(1,2)
@test typeof(tpb) == PB.TensorProductBasis{1,PB.LagrangePolynomialBasis{3,typeof(1.0)},3,typeof(1.0)}
@test allequal(tpb(-1.0),[1.0,0.0,0.0])
@test allequal(tpb(0.0),[0.0,1.0,0.0])
@test allequal(tpb(1.0),[0.0,0.0,1.0])
@test_throws AssertionError tpb([1.0,2.0])
@test allequal(tpb([1.0]),[0.0,0.0,1.0])
@test allequal(PB.gradient(tpb,-1.0),[-1.5,2,-0.5])
@test allequal(PB.gradient(tpb,0.0),[-0.5,0,0.5])
@test allequal(PB.gradient(tpb,1.0),[0.5,-2,1.5])
@test_throws AssertionError PB.gradient(tpb,[1.0,2.0])
@test allequal(PB.gradient(tpb,[1.0]),[0.5,-2,1.5])

@test_throws AssertionError PB.TensorProductBasis(1,-1)
@test_throws AssertionError PB.TensorProductBasis(1,2,start=2)
@test_throws AssertionError PB.TensorProductBasis(1,2,stop=-2)
tpb = PB.TensorProductBasis(1,2,stop=0.0)
@test allequal(tpb.points,[-1.0 -0.5 0.0])
@test allequal(tpb(-1.0),[1.0,0.0,0.0])
@test allequal(tpb(-0.5),[0.0,1.0,0.0])
@test allequal(tpb(0.0),[0.0,0.0,1.0])
@test_throws AssertionError tpb([1.0,2.0])
@test allequal(tpb([0.0]),[0.0,0.0,1.0])
@test allequal(PB.gradient(tpb,-1.0),[-3.,4,-1])
@test allequal(PB.gradient(tpb,-0.5),[-1.0,0,1])
@test allequal(PB.gradient(tpb,0.0),[1.0,-4,3])
@test_throws AssertionError PB.gradient(tpb,[1.0,2.0])
@test allequal(PB.gradient(tpb,[0.0]),[1.0,-4,3])

function test_basis_on_points(basis::PB.TensorProductBasis{2,T,N}) where {T,N}
    flag = true
    for i in 1:N
        vals = zeros(N)
        vals[i] = 1.0
        p = basis.points[:,i]
        flag = flag && basis(p[1],p[2]) ≈ vals
        flag = flag && basis(p...) ≈ vals
    end
    return flag
end

basis = PB.LagrangePolynomialBasis(0)
tp2 = PB.TensorProductBasis(2,basis)
@test allequal(tp2.points,[0.0,0.0])
@test test_basis_on_points(tp2)
@test_throws MethodError PB.gradient(tp2,1.5,1.0,2.0)
@test allequal(PB.gradient(tp2,1,0.0,0.0),[0.0])
@test allequal(PB.gradient(tp2,2,0.0,0.0),[0.0])
@test allequal(PB.gradient(tp2,2,[0.0,0.0]),[0.0])
@test allequal(PB.gradient(tp2,0.0,0.0),[0.0 0.0])
@test allequal(PB.gradient(tp2,[0.0,0.0]),[0.0 0.0])
@test_throws AssertionError PB.gradient(tp2,[0.0])
@test_throws AssertionError PB.gradient(tp2,[0.0,0.0,0.0])


v1 = [1.0,0.0,0.0]
v2 = [0.0,1.0,0.0]
v3 = [0.0,0.0,1.0]
d1 = [-1.5,2.0,-0.5]
d2 = [-0.5,0.0,0.5]
d3 = [0.5,-2.0,1.5]
tp2 = PB.TensorProductBasis(2,2)
@test_throws AssertionError PB.gradient(tp2,0,-1,1)
@test_throws AssertionError PB.gradient(tp2,3,-1,1)
@test allequal(PB.gradient(tp2, 1, -1.0, -1.0), kron(d1,v1))
@test allequal(PB.gradient(tp2, 1, -1.0, +0.0), kron(d1,v2))
@test allequal(PB.gradient(tp2, 1, -1.0, +1.0), kron(d1,v3))
@test allequal(PB.gradient(tp2, 1, +0.0, -1.0), kron(d2,v1))
@test allequal(PB.gradient(tp2, 1, +0.0, +0.0), kron(d2,v2))
@test allequal(PB.gradient(tp2, 1, +0.0, +1.0), kron(d2,v3))
@test allequal(PB.gradient(tp2, 1, +1.0, -1.0), kron(d3,v1))
@test allequal(PB.gradient(tp2, 1, +1.0, +0.0), kron(d3,v2))
@test allequal(PB.gradient(tp2, 1, +1.0, +1.0), kron(d3,v3))

@test allequal(PB.gradient(tp2, 1, [+1.0, +1.0]), kron(d3,v3))

@test allequal(PB.gradient(tp2, 2, -1.0, -1.0), kron(v1,d1))
@test allequal(PB.gradient(tp2, 2, -1.0, +0.0), kron(v1,d2))
@test allequal(PB.gradient(tp2, 2, -1.0, +1.0), kron(v1,d3))
@test allequal(PB.gradient(tp2, 2, +0.0, -1.0), kron(v2,d1))
@test allequal(PB.gradient(tp2, 2, +0.0, +0.0), kron(v2,d2))
@test allequal(PB.gradient(tp2, 2, +0.0, +1.0), kron(v2,d3))
@test allequal(PB.gradient(tp2, 2, +1.0, -1.0), kron(v3,d1))
@test allequal(PB.gradient(tp2, 2, +1.0, +0.0), kron(v3,d2))
@test allequal(PB.gradient(tp2, 2, +1.0, +1.0), kron(v3,d3))

@test allequal(PB.gradient(tp2, 2, [+1.0, +1.0]), kron(v3,d3))

@test allequal(PB.gradient(tp2,1.0,1.0),hcat(kron(d3,v3),kron(v3,d3)))
@test_throws AssertionError PB.gradient(tp2,[1.0,1.0,1.0])
@test allequal(PB.gradient(tp2,[1.0,1.0]),hcat(kron(d3,v3),kron(v3,d3)))

function test_basis_on_points(basis::PB.TensorProductBasis{3,T,N}) where {T,N}
    flag = true
    for i in 1:N
        vals = zeros(N)
        vals[i] = 1.0
        p = basis.points[:,i]
        flag = flag && basis(p[1],p[2],p[3]) ≈ vals
        flag = flag && basis(p) ≈ vals
    end
    return flag
end

tp3 = PB.TensorProductBasis(3,2)
@test test_basis_on_points(tp3)

@test_throws AssertionError PB.gradient(tp3,0,-1.0,-1.0,-1.0)
@test_throws AssertionError PB.gradient(tp3,4,-1.0,-1.0,-1.0)
@test allequal(PB.gradient(tp3,1,-1.0,-1.0,-1.0),kron(d1,v1,v1))
@test allequal(PB.gradient(tp3,2,-1.0,0.0,-1.0),kron(v1,d2,v1))
@test allequal(PB.gradient(tp3,3,-1.0,0.0,+1.0),kron(v1,v2,d3))

@test_throws AssertionError PB.gradient(tp3,1,[1.0])
@test_throws AssertionError PB.gradient(tp3,1,[1.0,1.0])
@test_throws AssertionError PB.gradient(tp3,1,[1.0,1.0,1.0,1.0])

@test allequal(PB.gradient(tp3,3,[-1.0,-1.0,-1.0]),kron(v1,v1,d1))

@test allequal(PB.gradient(tp3,1.0,-1.0,0.0),hcat(kron(d3,v1,v2),kron(v3,d1,v2),kron(v3,v1,d2)))
@test_throws AssertionError PB.gradient(tp3,[1.0])
@test_throws AssertionError PB.gradient(tp3,[1.0,2.0,1.0,3.0])
@test allequal(PB.gradient(tp3,[1.0,-1.0,0.0]),hcat(kron(d3,v1,v2),kron(v3,d1,v2),kron(v3,v1,d2)))


tp1 = PB.TensorProductBasis(1,2)
@test allequal(PB.hessian(tp1,-1),[1,-2,1])
@test allequal(PB.hessian(tp1,0),[1,-2,1])
@test allequal(PB.hessian(tp1,1),[1,-2,1])
@test allequal(PB.hessian(tp1,[-1]),[1,-2,1])
@test allequal(PB.hessian(tp1,[0]),[1,-2,1])
@test allequal(PB.hessian(tp1,[1]),[1,-2,1])


tp2 = PB.TensorProductBasis(2,2)
f(x) = x[1]^2 + 2x[1]*x[2] + x[2]^2
coeffs = mapslices(f,Array(tp2.points),dims=1)
@test all([allequal(coeffs*PB.hessian(tp2,tp2.points[:,i]),[2,2,2]) for i = 1:9])

tp3 = PB.TensorProductBasis(2,3)
f(x) = x[1]^3 + 2x[1]^2*x[2] + 18.0
tp3h(x) = [6x[1]+4x[2],4x[1],0.0]
coeffs = mapslices(f,Array(tp3.points),dims=1)
p = tp3.points
@test all([allequal(coeffs*PB.hessian(tp3,p[:,i]),tp3h(p[:,i]),1e3eps()) for i = 1:16])
