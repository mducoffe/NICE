#NICE model

"""
Non Linear independent components estimation and related classes
"""

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from blocks.bricks import Initializable, Sequence
from blocks.bricks.base import application
from mlp import TriangularMLP
from corruptor import Corruptor


pi=theano.shared(np.pi)
default_seed=[27,9,2014]

class Distribution(object):
    """
    Abstract class implementing methods related to distribution.
    """
    def __init__(self,rng=None, theano_rng=None, ** kwargs):
        super(Distribution, self).__init__(** kwargs)
        self.params = []
        if rng is None :
	  self.rng = np.random.RandomState(23455)
	else :
	  self.rng=rng
        if theano_rng is None:
	  self.theano_rng = RandomStreams(self.rng.randint(2**30))
	else :
	  self.theano_rng=theano_rng

    def get_log_likelihood(self, z):
        """
        Compute the log-likelihood of a batch according to this distribution.

        Parameters
        ----------
        z : tensor_like
            Point whose log-likelihood to estimate

        Returns
        -------
        log_likelihood : tensor_like
            Log-likelihood of z
        """
        raise NotImplementedError(str(type(self))
                                  + " does not implement log_p_z.")

    def sample(self, shape):
        """
        Sample from this distribution with the given shape.

        Parameters
        ----------
        shape : tuple
            Shape of the batch to sample

        Returns
        -------
        samples : tensor_like
            Samples
        """
        raise NotImplementedError(str(type(self))
                                  + " does not implement sample.")

    def entropy(self, shape):
        """
        Get entropy from this distribution with the given shape.

        Parameters
        ----------
        shape : tuple
            Shape of the batch to sample

        Returns
        -------
        samples : tensor_like
            Entropy
        """
        raise NotImplementedError(str(type(self))
                                  + " does not implement entropy.")

    def set_rng(self, rng):
        """
        SSetup the theano random generator for this class.

        Parameters
        ----------
        rng : np.random.RandomState
            Random generator from which to generate the seed of
            the theano random generator
        """
        
        self.rng = rng
        self.theano_rng = RandomStreams(self.rng.randint(2**30))
                                  
class StandardNormal(Distribution):
    
    def get_log_likelihood(self, Z):
        log_likelihood = -.5 * (T.sqr(Z) + sharedX(np.log(2 * np.pi)))
        log_likelihood = log_likelihood.sum(axis=-1)

        return log_likelihood

    
    def sample(self, shape):
        samples = self.theano_rng.normal(size=shape,
                                         dtype=theano.config.floatX)
        return samples
    
    def entropy(self, shape):
        entropy = .5 * sharedX(np.log(2 * np.pi) + 1.)
        entropy *= T.ones(shape)
        entropy = entropy.sum(axis=-1)

        return entropy


class StandardLogistic(Distribution):

    def get_log_likelihood(self, Z):
        log_likelihood = - (T.nnet.softplus(Z) + T.nnet.softplus(-Z))
        log_likelihood = log_likelihood.sum(axis=-1)

        return log_likelihood

    
    def sample(self, shape):
        samples = self.theano_rng.uniform(size=shape,dtype=theano.config.floatX)
        samples = T.log(samples) - T.log(1-samples)
        return samples
    
    
    def entropy(self, shape):
        entropy = 2.
        entropy *= T.ones(shape)
        entropy = entropy.sum(axis=-1)

        return entropy




class NICE(Sequence,Initializable,Distribution):
  
  def __init__(self, prior, encoder=None,
	  nvis=None, corruptor=None,
	  batch_size=None, **kwargs):
	    
    #we need to init the encoder !!!
    assert encoder is not None
    assert isinstance(encoder, TriangularMLP)
    self.encoder=encoder
    self.prior=prior
    super(NICE, self).__init__(application_methods=[self.encoder.apply], **kwargs)
    self.encoder.name = self.name+'_encoder'
    self.rng = np.random.RandomState(23455) 
    self.prior.set_rng(self.rng)    
    self.corruptor = corruptor

  def sparsity(self, alpha):
    return self.encoder.sparsity(alpha)

  def _push_allocation_config(self):
    self.encoder.push_allocation_config()
      
  def get_dim(self, name):
    return self.encoder.get_dim(name)
  
  def get_fprop_and_log_det_jacobian(self, X):
        """
        Get the state of the layer and the log-Jacobian determinant of the
        transformation.

        Parameters
        ----------
        X : tensor_like, member of self.input_space
            A minibatch of states of the layer below.

        Returns
        -------
        Z : tensor_like, member of self.output_space
            Code from the encoder.
        log_det_jac : tensor_like
            Log Jacobian determinant of the transformation.
        """
        Z, log_det_jac = X, 0.
        if self.encoder is not None:
            Z, log_det_jac = self.encoder.get_fprop_and_log_det_jacobian(X)
        return Z, log_det_jac 

  

  def get_log_likelihood(self, X):

     X_in = X
     # Corrupt the data if possible
     if self.corruptor is not None:
            X_in = self.corruptor.corrupt(X_in)

     Z, log_det_jac = self.get_fprop_and_log_det_jacobian(X_in)
     prior = self.log_p_z(Z)
     transformation_extension = log_det_jac

     log_likelihood = transformation_extension +prior
     return log_likelihood
  
                                
  def log_p_z(self, Z):
    """
        Compute the log-likelihood of a point according to
        the prior distribution.

        Parameters
        ----------
        z : tensor_like, member of self.output_space
            Code from the encoder

        Returns
        -------
        log_likelihood : tensor_like
            Log-likelihood of z for the prior distirbution
    """
    return self.prior.get_log_likelihood(Z)

  def decode(self, X):
	return self.encoder.inv_fprop(X)

  def sample(self, num_samples):
        """
        Sample from the model's learned distribution

        Parameters
        ----------
        num_samples : int
            Number of samples

        Returns
        -------
        samples : tuple of tensor_like
            Samples. The first element of the tuple is the actual sample, the
            others are intermediate quantities.
        """
        if isinstance(num_samples, tuple):
            shape = (num_samples[0], self.nvis)
        else:
            shape = (num_samples, self.nvis)
        Z = self.prior.sample(shape)

        samples = Z
        if self.encoder is not None:
            samples = self.encoder.inv_fprop(Z)
        return samples
    
  #@application(inputs=['input_'], outputs=['output'])    
  def encode(self, X):
        """
        Encode the data.

        Parameters
        ----------
        X : tensor_like, member of self.input_space
            Input

        Returns
        -------
        Z : tensor_like, member of self.output_space
            Code from the encoder
        """
        Z = X
        if self.encoder is not None:
            Z = self.encoder.apply(X)
        return Z

  @application(inputs=['input_'], outputs=['output'])    
  def apply(self, input_):
	return self.encode(input_)

    

        

                                     
    
