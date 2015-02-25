#class Corruptor

import numpy
import theano
import theano.tensor as T
# Shortcuts
RandomStreams = T.shared_randomstreams.RandomStreams
"""
theano.config.warn.sum_div_dimshuffle_bug = False
if 0:
  print 'WARNING: using SLOW rng'
  RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
  import theano.sandbox.rng_mrg
  RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams
"""
#faire une classe Corruptor ou il n'y a rien d'implementer
class Corruptor(object):
  
  def __init__(self, corruption_level, rng, **kwargs):
    super(Corruptor,self).__init__(**kwargs)
    self.corruption_level=corruption_level
    self.rng=rng
    self.theano_rng = RandomStreams(self.rng.randint(2**30))
    
  def corrupt(self,x):
    raise NotImplementedError(str(type(self))
                                  + " _corrupt(x)")

class Dequantizer(Corruptor): #instead of Corruptor rather RandomStreams
  """
  Dequantizer corruptor
  Adressing the log-likelihood problem of having arbitrarily high
  log-likelihood due to constant features of the data. Using Tapani Raiko's
  idea to ''dequantize'' the data. Corrupting in general put an upper bound
  on the log-likelihood of the data by the entropy of the corruption process.
  Parameters
  ----------
  low : float, optional
  Lowest value of the data
  high : float, optional
  Highest value of the data
  n_values : int, optional
  Number of quantum/values of the data
  """

  def __init__(self, corruption_level, rng, low=0., high=1., n_values=256, **kwargs):
    super(Dequantizer, self).__init__(corruption_level, rng, ** kwargs)
    assert high > low
    self.low = low
    self.high = high
    self.n_values = n_values
    
  def corrupt(self, x):
     #put the data between 0 and 1
     rval = x-self.low
     rval /=(self.high - self.low)
     #Add uniform noise to dequantize
     rval *= (self.n_values -1)
     rval +=self.corruption_level*self.theano_rng.uniform(size=x.shape,
	      dtype=theano.config.floatX )
     rval /= (self.n_values + self.corruption_level - 1)
     # Put back in the given interval
     rval *= (self.high - self.low)
     rval += self.low
     return rval
     
     
#test corruptor
if __name__ == '__main__':
  
  numpy_rng = numpy.random.RandomState(23455)
  corruptor = Dequantizer(0.1, numpy_rng)
  x=T.matrix()
  y=corruptor.corrupt(x)
  
  