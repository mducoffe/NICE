import numpy as np
import theano
import theano.tensor as T

from blocks.bricks import MLP, Activation, Identity, Initializable, Sequence, Feedforward
from blocks.bricks.base import application
from blocks.utils import shared_floatx_zeros
from blocks.roles import add_role, WEIGHTS, BIASES
from blocks.initialization import IsotropicGaussian


class Layer :
  
  @application(inputs=['input_'], outputs=['output'])
  def apply(self, input_):
     raise NotImplementedError(str(type(self))
                                  + " does not implement apply.")
                                  
  def inv_fprop(self, state):       
    raise NotImplementedError(str(type(self))
				  +"does not implement inv_fprop")
				  
  
  def get_fprop_and_log_det_jacobian(self, state_below):   
    raise NotImplementedError(str(type(self))
				  +"does not implement get_fprop_and_log_det_jacobian ")

  def sparsity(self, alpha):
     return 0.
	

class Homothety(Initializable,Feedforward, Layer):
  
  def __init__(self, dim, **kwargs):
    super(Homothety,self).__init__(**kwargs)
    assert type(dim) is int
    self.dim=dim
  
  @property
  def D(self):
    return self.params[0]

  
  def _allocate(self):
    D=shared_floatx_zeros((self.dim,), name='D')
    add_role(D, WEIGHTS)
    self.params.append(D)

  def _initialize(self):
    D, =self.params
    self.weights_init.initialize(D,self.rng)


  @application(inputs=['input_'], outputs=['output'])
  def apply(self, input_):
    D, = self.params
    state = input_ * T.exp(D).flatten()
    return state

  def get_dim(self, name):
    return self.dim

    
  def inv_fprop(self, state):
        """
        Inversion of the Homothety forward propagation.

        Parameters
        ----------
        state : tensor_like, member of self.output_space
            The state above the layer

        Returns
        -------
        state_below : tensor_like
            The resulting state below
        """
        D, = self.params
        state_below = state * T.exp(-D).flatten()

        return state_below

  def get_fprop_and_log_det_jacobian(self, state_below):
        """
        Get the state of the layer and the log-Jacobian determinant of the
        transformation.

        Parameters
        ----------
        state_below : tensor_like, member of self.input_space
            A minibatch of states below.

        Returns
        -------
        state : tensor_like, member of self.output_space
            A minibatch of states of this layer.
        log_det_jac : tensor_like
            Log Jacobian determinant of the transformation
        """
        D, = self.params
        return self.apply(state_below), self.D.sum()
        
class Reordering(Initializable, Layer):
    def __init__(self, dim, mode='tile', ** kwargs):
        super(Reordering, self).__init__(** kwargs)
        assert mode in ['tile', 'reverse']
        assert type(dim) is int
        self.dim = dim
        self.mode = mode
        half_dim = int(self.dim/2)
        self.permutation =np.arange(self.dim)
        if self.mode=='tile':
	  tmp=self.permutation.copy()
	  self.permutation[:half_dim]=tmp[::2]
	  self.permutation[-half_dim:] = tmp[1::2]
	elif self.mode =='reverse':
	  self.permutation = self.permutation[::-1]
	self.params=[]

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
      return input_[:,self.permutation]


    def inv_fprop(self, state):
        """
        Inversion of the Reordering forward propagation.

        Parameters
        ----------
        state : tensor_like, member of self.output_space
            The state above the layer

        Returns
        -------
        state_below : tensor_like
            The resulting state below
        """
        state_below = state[:, np.argsort(self.permutation)]

        return state_below

    def get_fprop_and_log_det_jacobian(self, state_below):
        """
        Get the state of the layer and the log-Jacobian determinant of the
        transformation.

        Parameters
        ----------
        state_below : tensor_like, member of self.input_space
            A minibatch of states below.

        Returns
        -------
        state : tensor_like, member of self.output_space
            A minibatch of states of this layer.
        log_det_jac : tensor_like
            Log Jacobian determinant of the transformation
        """
        return self.apply(state_below), 0.
        
      
class CouplingFunction(Sequence,Initializable,Layer):
  
  def __init__(self, dim, activations, split=0.5, reverse=True, **kwargs):
     self.activations = activations
     application_methods =[bricks.apply for bricks in activations]
     super(CouplingFunction, self).__init__(application_methods=application_methods,**kwargs)
     assert (split>0 and split <1) #percentage of the data
     self.split=split
     self.reverse=reverse
     self.dim = int(self.split*dim)
     self.dims=[self.dim for i in xrange(len(self.activations)+1)]
     self.permutation = T.arange(dim)
     if self.reverse:
      self.permutation= self.permutation[::-1]

  def _push_allocation_config(self):
    for function in self.activations:
      #check that function has push_allocation_config() as a method
      if hasattr(function,'push_allocation_config'):
	function.push_allocation_config()
      

  def get_dim(self, name):
    return dim

  def function(self, x):
    input_=x
    for layer in self.activations:
      result_temp = layer.apply(input_)
      input_=result_temp
    return input_


  @application(inputs=['input_'], outputs=['output'])
  def apply(self, input_):
    #first split the input
    #apply the coupling function
    # exchange data
    shape = input_.shape[1]
    #index = T.cast(T.round(self.split*shape),'int32')
    index = self.dim
    coupling_out = self.function(input_[:,:index])
    state = T.inc_subtensor(input_[:, index:], coupling_out)
    #define permutation
        
    state = state[:, self.permutation]
    return state
    
  def inv_fprop(self, output_):
	shape = output_.shape[1]
	index = self.dim
        state_below = output_
        state_below = state_below[:, T.argsort(self.permutation)]
        coupling_out = -self.function(state_below[:, :index])
        state_below = T.inc_subtensor(state_below[:, index:],
                                      coupling_out)

        return state_below
        
  def get_fprop_and_log_det_jacobian(self, state_below):
        log_det_jac = 0.

        return self.apply(state_below), log_det_jac   

class Branch(Sequence,Initializable,Layer):
  
  def __init__(self, dim, branch_to_rebuild=0, weights_init=IsotropicGaussian(0.01), **kwargs):
    self.branches=[Homothety(dim, weights_init=weights_init, use_bias=False) for i in xrange(2)] 
    application_methods =[bricks.apply for bricks in self.branches]
    assert type(dim) is int
    assert branch_to_rebuild>=0 and branch_to_rebuild<len(self.branches)
    self.dim=dim
    super(Branch,self).__init__(application_methods=application_methods, **kwargs) 
    self.branch_to_rebuild=branch_to_rebuild
    
  def _push_allocation_config(self):
    for branches in self.branches :
      branches.push_allocation_config()
      
  @application(inputs=['input_'], outputs=['output'])
  def apply(self, input_):
     return self.branches[0].apply(input_) #+ self.branches[1].apply(input_) #TODO : more branches, pas de code en dur
                                  
  def inv_fprop(self, state):       
    return self.branches[self.branch_to_rebuild].inv_fprop(state)
				  
  
  def get_fprop_and_log_det_jacobian(self, state_below):   
    D0, = self.branches[0].params
    D1, = self.branches[1].params
    return self.apply(state_below), (D0+D1).sum()

  def set_branch_to_rebuild(index):
    assert index >=0 and index <len(self.branches)
    self.branch_to_rebuild=index
  
  @application(inputs=['input_'], outputs=['output'])
  def apply_branch(self, input_):
    return self.branches[self.branch_to_rebuild].apply(input_)
    
  def sparsity(self, alpha):
    D, =self.branches[self.branch_to_rebuild].params
    return alpha*(D**2).sum()
