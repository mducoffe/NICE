#define branch

import numpy as np
import theano
import theano.tensor as T

from blocks.bricks import MLP, Activation, Identity, Initializable, Sequence, Feedforward
from blocks.bricks.base import application
from blocks.utils import shared_floatx_zeros
from blocks.roles import add_role, WEIGHTS, BIASES

class Branch(Initializable, Feedforward, Layer):
  
  def __init__(self, dim, branch_to_rebuild=0, weights_init=IsotropicGaussian(0.01), **kwargs):
    self.branches=[Homothety(dim, weights_init=weights_init, use_bias=False) for i in xrange(2)] #TODO
    application_methods =[self.bricks.apply for bricks in self.branches]
    assert type(dim) is int
    assert branch_to_rebuild>0 and branch_to_rebuild<len(self.branches)
    self.dim=dim
    super(Branch,self).__init__(application_methods=application_methods, **kwargs) 
    self.branch_to_rebuild=branch_to_rebuild
    
  def _push_allocation_config(self):
    for branches in self.branches :
      branches.push_allocation_config()
      
  @application(inputs=['input_'], outputs=['output'])
  def apply(self, input_):
     return T.sum([branch.apply(input) for branch in self.branches])
                                  
  def inv_fprop(self, state):       
    return self.branches[self.branch_to_rebuild].inv_fprop(state)
				  
  
  def get_fprop_and_log_det_jacobian(self, state_below):   
    D, = T.sum([branch.params for branch in self.branches], axis=0)
    return self.apply(state_below), self.D.sum()

  def set_branch_to_rebuild(index):
    assert index >0 and index <len(self.branches)
    self.branch_to_rebuild=index
  
  @application(inputs=['input_'], outputs=['output'])
  def apply_branch(self, input_):
    return self.branches[self.branch_to_rebuild].apply(input_)