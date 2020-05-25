using Test
using Revise
using PolynomialBasis

PB = PolynomialBasis

function allequal(v1,v2)
    return all(v1 .≈ v2)
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
tp = PB.tensor_product_points(1,p)
@test allequal(p,tp)

tp = PB.tensor_product_points(2,p)
testtp = [1.0;1.0]
@test allequal(tp,testtp)

tp = PB.tensor_product_points(3,p)
testtp = [1.0;1.0;1.0]
@test allequal(tp,testtp)

basis = PB.LagrangePolynomialBasis(0)
@test_throws AssertionError PB.TensorProductBasis(0,basis)
@test_throws AssertionError PB.TensorProductBasis(4,basis)
tpb = PB.TensorProductBasis(1,basis)
@test typeof(tpb) == PB.TensorProductBasis{1,PB.LagrangePolynomialBasis{1},1}
@test allequal(tpb(0.0),[1.0])
@test allequal(PB.gradient(tpb,0.0),[0.0])
@test allequal(PB.gradient(tpb,[0.0]),[0.0])
@test_throws AssertionError PB.gradient(tpb,[0.0,0.0])

tpb = PB.TensorProductBasis(1,0)
@test typeof(tpb) == PB.TensorProductBasis{1,PB.LagrangePolynomialBasis{1},1}
@test allequal(tpb(0.0),[1.0])
@test allequal(PB.gradient(tpb,0.0),[0.0])
@test allequal(PB.gradient(tpb,[0.0]),[0.0])
@test_throws AssertionError PB.gradient(tpb,[0.0,0.0])

tpb = PB.TensorProductBasis(1,1)
@test typeof(tpb) == PB.TensorProductBasis{1,PB.LagrangePolynomialBasis{2},2}
@test allequal(tpb(-1.0),[1.0,0.0])
@test allequal(tpb(1.0),[0.0,1.0])
@test_throws AssertionError tpb([1.0,2.0])
@test allequal(tpb([1.0]),[0.0,1.0])
@test allequal(PB.gradient(tpb,-1.0),[-0.5,0.5])
@test allequal(PB.gradient(tpb,1.0),[-0.5,0.5])
@test_throws AssertionError PB.gradient(tpb,[1.0,2.0])
@test allequal(PB.gradient(tpb,[1.0]),[-0.5,0.5])

tpb = PB.TensorProductBasis(1,2)
@test typeof(tpb) == PB.TensorProductBasis{1,PB.LagrangePolynomialBasis{3},3}
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
