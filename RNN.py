import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer

class RNN(object):
    def __init__(self, n_in, n_out, width, depth, activation = tf.tanh, 
                 random_seed = None):
        
        self.width = width
        self.n_in = n_in
        self.n_out = n_out
        self.depth = depth
        self.activation = activation
        self.random_seed = random_seed
        
        xavier_init = xavier_initializer(seed = random_seed)
        params = []
        
        self.init_state=tf.Variable(xavier_init([1,width]),name="init_state")
        self.Wxh = tf.Variable(xavier_init([n_in,width]),name="Wxh")
        self.Whh = tf.Variable(xavier_init([width,width]),name="Whh")
        self.Why = tf.Variable(xavier_init([width,n_out]),name="init_state")
        self.bh  = tf.Variable(xavier_init([width]),name="bh")
        self.by  = tf.Variable(xavier_init([n_out]),name="by")

        
#         params = list(self.init_state.values(),self.Wxh.values(),self.Whh.values(),self.Why.values(),self.bh.values(),self.by.values())
 
        self.params = params
        
        

         
    def __call__(self, x, linear_out = True):
        state=self.activation(x@ self.Wxh + self.init_state@ self.Whh + self.bh)
        
        for i in range(self.depth):
            state=self.activation(x@ self.Wxh + state@ self.Whh + self.bh)
        
        y=state@self.Why + self.by
        if not (linear_out):
                y = self.activation(y)
        return(y)

            
            