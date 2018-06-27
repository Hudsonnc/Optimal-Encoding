import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import lib
from tqdm import tqdm

EPS = 1e-8

class OptimalEncoding(object):
    def __init__(self, encoder, decoder, k):
        #Encoder n_out must equal k
        self.encoder = encoder
        #Decoder n_in must equal k
        self.decoder = decoder
        
        self.params = encoder.params + decoder.params
        
        self.X_dim = encoder.n_in
        self.Y_dim = decoder.n_out - 1 # extra dimension for model (spherical) variance
        self.k = k
        
        #Data
        self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, self.Y_dim], name='Y')
        
        #Encode
        self.f_X = encoder(self.X, linear_out=True)
        #placeholder for epsilon
        self.epsilon = tf.placeholder(tf.float32, shape=(None, self.k), name = 'epsilon')
        #placeholder for sigma
        self.sigma = tf.placeholder(tf.float32, shape=(), name = 'sigma')
        #Add noise to encoding
        self.Z = self.f_X + self.sigma * self.epsilon
        
        #Decode
        self.Y_hat = decoder(self.Z, linear_out=True)
        self.std = self.Y_hat[:,-1]
        self.Y_hat = self.Y_hat[:,:-1]
        
        #Losses 
        #Autoencoder 
        self.L1 = tf.reduce_mean(
            tf.reduce_sum(
                tf.abs(self.Y - tf.tanh(self.Y_hat))/tf.expand_dims((tf.abs(self.std) + EPS), 1) + tf.expand_dims(tf.log(tf.abs(self.std) + EPS), 1),
                axis=1
            )
        )
        
        #Classification
        self.CrossEnt = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(self.Y), logits=self.Y_hat)
        )
        
        #Stein entropy
        self.Entropy = tf.reduce_mean(tf.einsum('ij,ij->i', self.Z, tf.stop_gradient(lib.stein_d_H(self.Z))))
        
    def train(self, x, y=None, min_entropy=True, epochs=100, batch_size=64, lr=1e-3, sigma = 1, task = 'autoencoder'):
        taskdict = {
            'autoencoder': self.L1,
            'classification': self.CrossEnt
        }
        
        y = (x if y is None else y)
        
        if task in taskdict:
            self.taskLoss = taskdict[task]
            self.Loss = self.taskLoss + (self.Entropy if min_entropy else tf.stop_gradient(self.Entropy))
        else:
            raise ValueError('task not supported yet')

        #Optimizer 
        solver = tf.train.AdagradOptimizer(learning_rate = lr).minimize(self.Loss, var_list=self.params)
        
        
        #Training
        init = tf.global_variables_initializer()
        sess = tf.Session()
        self.sess = sess
        with sess.as_default():
            sess.run(init)

            losses = []
            tasklosses = []
            ents = []
            
            n_batches = int(x.shape[0]/float(batch_size))
            for epoch in tqdm(range(epochs)):
                rand_idxs = np.arange(x.shape[0]) 
                np.random.shuffle(rand_idxs)
                
                loss = 0
                task_loss = 0
                ent = 0
                for batch in range(n_batches):
                    mb_idx = rand_idxs[batch*batch_size:(batch+1)*batch_size]
                    x_mb = x[mb_idx]
                    y_mb = y[mb_idx]
                    
                    epsilon = np.random.normal(0,1, size=(batch_size, self.k))
                    
                    _, loss_curr, taskloss_curr, ent_curr = sess.run(
                        [
                            solver, self.Loss, self.taskLoss, self.Entropy,
                        ], 
                        feed_dict = {self.X:x_mb, self.Y:y_mb, self.epsilon:epsilon, self.sigma:sigma})
                    
                    loss += loss_curr/n_batches
                    task_loss += taskloss_curr/n_batches
                    ent += ent_curr/n_batches
                    
                losses.append(loss)
                tasklosses.append(task_loss)
                ents.append(ent)

            print('Final task loss: %f' %(tasklosses[-1]))
            
            plt.figure()
            plt.plot(losses)
            plt.title('total loss')
            
            plt.figure()
            plt.plot(tasklosses)
            plt.title('task loss')
            
            plt.figure()
            plt.plot(np.array(ents))
            plt.title('ent')

    def encode(self, x, sigma = 1):
        epsilon = np.random.normal(0,1, size=(len(x), self.k))
        return(self.sess.run(self.Z, feed_dict = {self.X:x, self.epsilon:epsilon, self.sigma:sigma}))
    
    def decode(self, z):
        return(self.sess.run(self.decoder(z, linear_out=True))[:,:-1])