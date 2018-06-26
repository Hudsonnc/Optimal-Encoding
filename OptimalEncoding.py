import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import lib

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
        
        #Encode for reparametrization trick
        self.Z_params = encoder(self.X, linear_out=True)
        
        #Placeholder for temperature
        self.tau = tf.placeholder(tf.float32, shape=(), name = 'tau')
        
        #Gumbel reparametrization
        self.probs = tf.sigmoid(self.Z_params)
        self.pis = tf.stack((self.probs, 1-self.probs), axis=2)
        self.g = tf.placeholder(tf.float32, shape=self.pis.shape, name='g')
        self.Z = lib.gumbel_soft(self.pis, self.g, tau = self.tau)[:,:,0]
        
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
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(self.Y), logits=self.Y_hat)/(tf.square(self.std) + EPS) 
            + tf.log(tf.abs(self.std) + EPS)
        )
        
        #Encoder entropy
        #Column wise entropy
        p = tf.reduce_mean(self.probs, 0)
        self.Entropy = -tf.reduce_sum(p * tf.log(p + EPS) + (1-p) * tf.log(1-p + EPS)) #in nats
        
    def train(self, x, y=None, max_entropy=True, epochs=100, batch_size=64, lr=1e-3, tau_rate = 1e-4, task = 'autoencoder'):
        taskdict = {
            'autoencoder': self.L1,
            'classification': self.CrossEnt
        }
        
        schedule = lambda i: np.float32(np.max((0.5, np.exp(-tau_rate*i))))
        
        y = (x if y is None else y)
        
        if task in taskdict:
            self.taskLoss = taskdict[task]
            self.Loss = self.taskLoss - (self.Entropy if max_entropy else tf.stop_gradient(self.Entropy))
        else:
            raise ValueError('task not supported yet')

        #Optimizer 
        solver = tf.train.AdagradOptimizer(learning_rate = lr).minimize(self.Loss, var_list=self.params)
        #Need to clip gradients as they get huge for gumbel softmax + stein gd
        #gradients, variables = zip(*solver.compute_gradients(self.Loss, var_list=self.params))
        #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        #solver = solver.apply_gradients(zip(gradients, variables))

        #Training
        init = tf.global_variables_initializer()
        sess = tf.Session()
        self.sess = sess
        with sess.as_default():
            sess.run(init)

            losses = []
            tasklosses = []
            ents = []
            stds = []
            
            n_batches = int(x.shape[0]/float(batch_size))
            for epoch in range(epochs):
                rand_idxs = np.arange(x.shape[0]) 
                np.random.shuffle(rand_idxs)
                
                loss = 0
                task_loss = 0
                ent = 0
                std = 0
                for batch in range(n_batches):
                    tau = schedule(epoch*n_batches + batch)

                    mb_idx = rand_idxs[batch*batch_size:(batch+1)*batch_size]
                    x_mb = x[mb_idx]
                    y_mb = y[mb_idx]

                    g = np.random.gumbel(size=(len(x_mb), self.k, 2))
                    _, loss_curr, taskloss_curr, ent_curr, std_curr = sess.run([solver, self.Loss, self.taskLoss, self.Entropy, self.std], feed_dict = {self.X:x_mb, self.Y:y_mb, self.g: g, self.tau: tau})
                    
                    loss += loss_curr/n_batches
                    task_loss += taskloss_curr/n_batches
                    ent += ent_curr/n_batches
                    std += np.mean(np.abs(std_curr))/n_batches
                    
                losses.append(loss)
                tasklosses.append(task_loss)
                ents.append(ent)
                stds.append(std)

            print('Final task loss: %f' %(tasklosses[-1]))
            
            plt.figure()
            plt.plot(losses)
            plt.title('total loss')
            
            plt.figure()
            plt.plot(tasklosses)
            plt.title('task loss')
            
            plt.figure()
            plt.plot(np.array(ents) * 1.442695) #in bits
            plt.title('ent')
            
            plt.figure()
            plt.plot(stds)
            plt.title('std')
            

    def encode(self, x, tau = 1e-10):
        g = np.random.gumbel(size=(len(x), self.k, 2))
        return(self.sess.run(self.Z, feed_dict = {self.X:x, self.g: g, self.tau: tau}))
    
    def decode(self, z):
        return(self.sess.run(self.decoder(z, linear_out=True)[:,:-1]))