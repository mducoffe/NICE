import theano.tensor as T
import theano
import numpy
import sys
from mlp import TriangularMLP
from layer import CouplingFunction, Homothety, Reordering, Branch
from corruptor import Dequantizer
from model import StandardLogistic, NICE
from blocks.bricks import MLP, Rectifier, Linear, Tanh, Softmax
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.cost import Cost, CategoricalCrossEntropy, MisclassificationRate
from blocks.bricks.base import application
from blocks.datasets.mnist import MNIST
from blocks.datasets.streams import DataStream
from blocks.datasets.schemes import SequentialScheme
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.roles import WEIGHTS

from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter


from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing

from blocks import config
config.data_path = 'data'


def build_nice(dimension):
  activations=[]
  activations.append(Reordering(dimension));
  
  for i in xrange(3):
    temp=[Rectifier() for j in xrange(5)]
    
    temp.append(Linear(name='output_'+str(i), input_dim=dimension/2, output_dim=dimension/2,weights_init=IsotropicGaussian(0.01),
			use_bias=False))
    
    #create Coupling Function
    cpl = CouplingFunction(dimension,temp,0.5)
    activations.append(cpl)
  
  #activations.append(Homothety(dimension,weights_init=IsotropicGaussian(0.01)))
  activations.append(Branch(dimension))
  #test MLP
  encoder = TriangularMLP(activations, dimension)
  #build prior
  prior = StandardLogistic()
  corruptor=Dequantizer(1, numpy.random.RandomState(23455))
  nice= NICE(prior, encoder=encoder, nvis=dimension,corruptor=corruptor)
  nice.initialize()
  return nice
  
def build_classifier(dimension):
  mlp = MLP([Tanh(),Tanh(), Softmax()], [784, 100,50, 10],
          weights_init=IsotropicGaussian(0.01),
          biases_init=Constant(0))
          
  mlp.initialize()
  return mlp
  
class Log_Likelihood(Cost):
  
  @application(outputs=["log_likelihood"])
  def apply(self, y_hat):
    
    return -T.cast(y_hat.mean(), theano.config.floatX)
    
if __name__ == '__main__':
  
  print 'Test NICE on blocks'
  print 'Step 1 : build MNIST NICE model'

  dimension=784
  nice = build_nice(dimension)
  mlp=build_classifier(dimension)
  x=T.matrix('features')
  y_hat_A = nice.get_log_likelihood(x)
  cost_A = -T.cast(y_hat_A.mean(), theano.config.floatX)
  
  #supervised part
  y = T.lmatrix('targets')
  y_hat_B = mlp.apply(nice.apply(x))
  cost_B = CategoricalCrossEntropy().apply(y.flatten(), y_hat_B)
  error_rate = MisclassificationRate().apply(y.flatten(), y_hat_B)
  
  cg = ComputationGraph([cost_B])
  liste = VariableFilter(roles=[WEIGHTS])(cg.variables)
  print (liste[0], liste[1], liste[2])
  
  
  #sparsity on the image information
  cost_C= 0.0005*T.sum([(liste[i]**2).sum() for i in xrange(3)])
  
  
  cost=cost_A+5*cost_B
  cost.name='cost'

  print "load dataset"
  mnist = MNIST("train")
  #mnist.sources=('features',)
 
  
  data_stream = DataStream(mnist, iteration_scheme=SequentialScheme(
     num_examples=mnist.num_examples, batch_size=200))
  
  
  
  algorithm = GradientDescent(cost=cost, step_rule=Scale(learning_rate=1e-3))   
  
  mnist_test = MNIST("test")
  #mnist_test.sources=('features',)
  data_stream_test = DataStream(mnist_test, iteration_scheme=SequentialScheme(
     num_examples=mnist_test.num_examples, batch_size=200))
  monitor = DataStreamMonitoring(
     variables=[error_rate], data_stream=data_stream_test, prefix="test")
     
  main_loop = MainLoop(model=None, data_stream=data_stream, algorithm=algorithm,
                      extensions=[monitor,FinishAfter(after_n_epochs=100), Printing()])
  
  main_loop.run()
  
  
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
  
