#test NICE
#Layer + MLP
import numpy
import theano.tensor as T
import theano
from blocks.bricks import Rectifier, Linear
from blocks.initialization import IsotropicGaussian
from layer import Reordering, Homothety, CouplingFunction
from mlp import TriangularMLP

if __name__ == '__main__':
  print 'kikou'
  dimension=10
  x=theano.shared(numpy.random.ranf((1, dimension)))
  temp=[Rectifier()]
  obj = Linear(input_dim=dimension/2, output_dim=dimension/2, name='linear2',weights_init=IsotropicGaussian(0.01), use_bias=False)
  temp.append(obj)
  cpl = CouplingFunction(dimension, temp)
  f1 = cpl
  f1.initialize()
  y=f1.apply(x)
  x_hat=f1.inv_fprop(y)
  
  print x.eval()
  print x_hat.eval()
  """
  f1 = Reordering(dimension)
  f2 = Homothety(dimension,weights_init=IsotropicGaussian(0.01))
  temp=[Rectifier()]
  obj = Linear(input_dim=dimension/2, output_dim=dimension/2, name='linear2',weights_init=IsotropicGaussian(0.01), use_bias=False)
  temp.append(obj)
  cpl = CouplingFunction(dimension, temp)
  
  mlp=TriangularMLP([f1,cpl,f2],dimension)
  
  mlp.initialize()
  y1=mlp.apply(x)
  z1,b1=mlp.get_fprop_and_log_det_jacobian(y1)
  b2 = mlp.inv_fprop(y1)
  print x.eval()
  print y1.eval()
  print z1.eval()
  print b1.eval()
  print b2.eval()
  """
  