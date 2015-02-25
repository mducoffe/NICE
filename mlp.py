import numpy
from theano import tensor
from blocks.bricks.base import application,lazy
from blocks.bricks import Sequence, Initializable, Feedforward
from layer import Layer

class TriangularMLP(Sequence, Initializable, Feedforward):
  
  @lazy
  def __init__(self, layers,dim, **kwargs):
    self.layers=layers
    self.dim = dim
    #nom des layers
    #self.linear_transformations = [Linear(name='linear_{}'.format(i))
    application_methods = [brick.apply for brick in
        self.layers if brick is not None]
    super(TriangularMLP,self).__init__(application_methods, **kwargs)
    
    #check that each layer is an instance of Layer -> right function
    for layer in self.layers:
      assert isinstance(layer, Layer)
  
  def sparsity(self, alpha):
    return self.layers[-1].sparsity(alpha)
    
  @property
  def input_dim(self):
    return self.dim
    
  @property
  def output_dim(self):
    return self.dim
    
  """
  @application(inputs=['input_'], outputs=['output'])
  def apply(self, input_):
    x=input_
    for layer in self.layers :
      result_temp = layer.apply(x)
      x=result_temp
      
    return x
  """ 
  def _push_allocation_config(self):
    for layer in self.layers:
      layer.push_allocation_config()
    """
    for attr in ['filter_size', 'num_filters', 'num_channels', 'image_size',
	'step', 'border_mode']:
	  setattr(self.convolution, attr, getattr(self, attr))
    self.convolution._push_allocation_config()
    """
  def inv_fprop(self, state, return_all=False):
        """
        Inversion of the MLP forward propagation.

        Parameters
        ----------
        state : tensor_like, member of self.output_space
            The state above the MLP

        Returns
        -------
        state_below : tensor_like
            The resulting state below
        """
        state_below = state

        if return_all:
            state_below = [state_below]

        for layer in self.layers[::-1]:
            if return_all:
                state_below.append(layer.inv_fprop(state_below[-1]))
            else:
                state_below = layer.inv_fprop(state_below)

        return state_below
  
  def get_fprop_and_log_det_jacobian(self, state_below):
        """
        Get the state of the MLP and the log-Jacobian determinant of the
        transformation.

        Parameters
        ----------
        state_below : tensor_like, member of self.input_space
            A minibatch of states below.

        Returns
        -------
        state : tensor_like, member of self.output_space
            A minibatch of states of this MLP.
        log_det_jac : tensor_like
            Log Jacobian determinant of the transformation
        """
        state = state_below
        log_det_jac = 0.
        for layer in self.layers:
            state, log_det_jac_layer = layer.get_fprop_and_log_det_jacobian(
                state
            )
            log_det_jac += log_det_jac_layer

        return state, log_det_jac  
        




 
