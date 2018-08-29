import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import lib
from tqdm import tqdm

EPS = 1e-8

import sys
sys.path.append('/home/AD/lbreston/NPEET_LNC/')
from lnc import MI
entropy = lambda x: MI.entropy(x,k=3,base=np.exp(1),intens=1e-10)

class OptimalEncoding(object):
    def __init__(self, encoder, decoder, k, activation = tf.tanh):
        if activation is None:
            activation = lambda x: x
            
        #Encoder n_out must equal k
        self.encoder = encoder
        #Decoder n_in must equal k
        self.decoder = decoder
        
        self.params = encoder.params + decoder.params
        
        self.X_dim = encoder.n_in
        self.Y_dim = decoder.n_out
        self.k = k
        
        #Data
        self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, self.Y_dim], name='Y')
        
        #Encode
        self.f_X = encoder(self.X, linear_out=True)
        #placeholder for n_samples in latent space
        self.n_samples = tf.placeholder(tf.int32, shape = (), name = 'n_samples')
        #placeholder for epsilon
        self.epsilon = tf.placeholder(tf.float32, shape=(None, None, self.k), name = 'epsilon')
        #placeholder for sigma
        self.sigma = tf.placeholder(tf.float32, shape=(), name = 'sigma')
        #Add noise to encoding
        self.Z = tf.expand_dims(self.f_X, 1)*(1+self.sigma * self.epsilon)
        
        #Decode: reshape Z to (None*n_samples, k) to decode and then reshape back to (None, n_samples, y_dim) 
        self.Y_hat = tf.reshape(
            decoder(
                tf.reshape(
                    self.Z,
                    tf.stack([-1, self.k])
                ),
                linear_out=True
            ),
            tf.stack([-1, self.n_samples, self.Y_dim])
        )
        
        #Losses 
        #Autoencoder 
        self.Eltwise_Deviation = tf.abs(tf.expand_dims(self.Y, 1) - activation(self.Y_hat))
        self.Laplace_Homoskedastic = tf.reduce_sum(#Sum result over features to get scalar (diagonal covariance)
            tf.log(#log of mean features across all batchsize * n_samples (homoskedastic in columns)
                tf.reduce_mean(#Mean across batchsize, n_samples.
                    self.Eltwise_Deviation, 
                    axis=(0,1)
                ) + EPS
            )
        )
        self.Laplace_Heteroskedastic = tf.reduce_mean(#Mean over batch_size to get scalar
            tf.reduce_sum(#sum over features (diagonal covariance)
                tf.log(#log of mean features across n_samples
                    tf.reduce_mean(#mean across n_samples
                        self.Eltwise_Deviation,
                        axis=(1)
                    ) + EPS
                ),
                axis=(-1)
            )
        )
        
        #Classification
        self.Samplewise_CrossEnt = tf.einsum(
           'ik,ijk->ij',
           self.Y, 
            -tf.log(tf.exp(self.Y_hat) / (tf.expand_dims(tf.reduce_sum(tf.exp(self.Y_hat), axis=-1), axis=-1) + EPS) + EPS)
        )
        self.CrossEnt_Homoskedastic = .5*tf.log(
            tf.reduce_mean(#Mean over batchsize, n_samples 
                self.Samplewise_CrossEnt
            ) + EPS
        )
        self.CrossEnt_Heteroskedastic = .5*tf.reduce_mean( #Mean over batchsize
            tf.log(
                tf.reduce_mean(#Mean over n_samples
                    self.Samplewise_CrossEnt,
                    axis = (1)
                ) + EPS
            )
        )
        
        #Stein entropy
        flat_Z = tf.reshape(
                    self.Z,
                    tf.stack([-1, self.k])
                )
        self.Entropy =  tf.reduce_mean(tf.einsum('ij,ij->i', flat_Z, tf.stop_gradient(lib.stein_d_H(flat_Z, -1))))
        
    def train(self, x, x_val, y_val, y=None, min_entropy=True, epochs=100, batch_size=64, lr=1e-3, n_samples = 1, sigma = 1, task = 'autoencoder', heteroskedastic = False):
        
        self.Laplace = self.Laplace_Heteroskedastic if heteroskedastic else self.Laplace_Homoskedastic
        self.CrossEnt = self.CrossEnt_Heteroskedastic if heteroskedastic else self.CrossEnt_Homoskedastic
        
        taskdict = {
            'autoencoder': self.Laplace,
            'classification': self.CrossEnt
        }
        
        y = (x if y is None else y)
        
        
        
        if task in taskdict:
            self.taskLoss = taskdict[task]
            self.Loss =  (self.Entropy if min_entropy else tf.stop_gradient(self.Entropy)) + self.taskLoss 
        else:
            raise ValueError('task not supported yet')

        #Optimizer 
        solver = tf.train.RMSPropOptimizer(learning_rate = lr).minimize(self.Loss, var_list=self.params)
        
        #Training
        init = tf.global_variables_initializer()
        sess = tf.Session()
        self.sess = sess
        with sess.as_default():
            sess.run(init)

            losses = []
            val_accs = []
            train_accs = []
            tasklosses = []
            ents = []
            knn_ents = []
            
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
                    
                    epsilon = np.random.normal(0,1, size=(batch_size, n_samples, self.k))
                    
                    _, loss_curr, taskloss_curr, ent_curr = sess.run(
                        [
                            solver, self.Loss, self.taskLoss, self.Entropy
                        ], 
                        feed_dict = {self.X:x_mb, self.Y:y_mb, self.epsilon:epsilon, self.sigma:sigma, self.n_samples: n_samples})
                    
                    loss += loss_curr/n_batches
                    task_loss += taskloss_curr/n_batches
                    ent += ent_curr/n_batches
                    
                zhat = self.encode(x_mb, sigma=sigma)
                knn_ent = entropy(zhat)
                
                if epoch%1000==0:
                    zhat_train = self.encode(x, sigma=sigma)
                    pred_train = self.decode(zhat_train)
                    pred_train = np.argmax(pred_train,1)
                    true_train = np.argmax(y,1)
                    train_acc = np.mean(pred_train == true_train)


                    zhat_val = self.encode(x_val, sigma=sigma)
                    pred_val = self.decode(zhat_val)
                    pred_val = np.argmax(pred_val,1)
                    true_val = np.argmax(y_val,1)
                    val_acc = np.mean(pred_val == true_val)
                    
                    train_accs.append(train_acc)
                    val_accs.append(val_acc)
                               
                losses.append(loss)
                tasklosses.append(task_loss)
                ents.append(ent)
                knn_ents.append(knn_ent)
            
            print('Sigma: %f' %(sigma))

            print('Final task loss: %f' %(tasklosses[-1]))
            
            plt.figure()
            plt.plot(losses)
            plt.title('total loss')
            
            
            plt.figure()
            plt.plot(tasklosses)
            plt.title('task loss')
            
            
            plt.figure()
            plt.plot(np.array(ents))
            plt.title('pseudo entropy loss')
            
            
            plt.figure()
            plt.title('knn estimated entropy')
            plt.plot(knn_ents)
            

            plt.figure()
            plt.title('train and validation accuracy')
            plt.plot(train_accs)
            plt.plot(val_accs)
           
            

    def encode(self, x, sigma = 0):
        epsilon = np.random.normal(0,1, size=(len(x), 1, self.k))
        return(self.sess.run(self.Z, feed_dict = {self.X:x, self.epsilon:epsilon, self.sigma:sigma, self.n_samples: 1})[:,0,:])
    
    def decode(self, z):
        return(self.sess.run(self.decoder(z, linear_out=True)))
    

