#restricted boltzmann machine (RBM)
#shallow neural network (2 layer neural network) that learn to reconstruct data by themselves in unspuervised fashion
#neuron at same layer are disconnected
#parameter to enhance weight and bias is loss (rather than cost)
#3 actions
#1. Forward Pass- all input is passed to the hidden layer
#2. Activation function trigger (or not) based on input
#3. Backward Pass - the data from hidden layer  is  passed to the input layer backward (depending on which neuron of hidden layer were activated, only from those neuron the data will go

#RBM take an input, translate them to a set of numbers that represent them.->these numbers can be translated back to reconstruct the input

#used to reconstruct some unstructred data (like image or video)->it auto-encodes

#Can automatically extract meaningful features from the input
#collaborative filtering, dimensionality reduction
#classification, regression, feature learning, topic modelling

#Generative vs Discriminative Model
#Discriminative Model: using activation function(logistic regression) etc it creates a decision boundary between classes (discriminates)

#Generative: for eg: classify cars whether suv or sedan?
	#generative builds a model of what suv car look like
	#what sedan cars look like
	#finally, any new input is matched against sedan model and suv model, to see whether new car looks more like suv or sedan

#Generative model, specify probability distribution over a dataset of input vectors.
#In unsupervised task->form a model for P(x); x:input vector
#Supervised task->form a model for P(x|y); y:label for x
#eg.if y indicates whether car suv(0) or sedan(1)
#p(x|y=0) models distribution of SUVs' features
#p(x|y=1) models distribution of sedans' features

#build generative model, create synthetic data by directly sampling from modelled probability distributions




#initialization

#load utility file containing utility functions which will help in processing output
import urllib
response = urllib.urlopen('http://deeplearning.net/tutorial/code/utils.py')
content = response.read()
target = open('utils.py', 'w')
target.close()

#load packages
import numpy as np
import tensorflow as tf

#load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data

from PIL import Image
from utils import tile_raster_images
import matplotlib.pyplot as plt

#load mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


#RBM layers; 1st->visible layer
#Mnist has 784 pixels (28x28)

#second layer: hidden layer possesses i neurons (here 500)
#hidden unit has binary state (si) either on or off, with probability depending on activation function(logistic function) of the inputs it receives from j visible units

#each node of first layer has a bias (vb)->visible bias
#bias of second layer->hb

vb = tf.placeholder("float", [784])
hb = tf.placeholder("float", [500])

#weights
#no. of rows->input nodes
#no. of columns->output nodes (hidden layer nodes)

#here, W->784x500 (784: visible neurons, 500 hidden neurons)
W = tf.placeholder("float", [784, 500])

#RBM assign value to "binary state vector"(here 1 to 500) over it's visible units
#possible confiurations: 2^n (n:no.of visible neurons)

#training
#phases: 1. forward pass	2. backward pass





#1. forward pass: processing in each node in hidden layer
#input from all visible nodes, passed to all hidden node
#at hidden layer:operation on X-> X*W + h_bias
	#result fed to sigmoid function->produce node's output/state

#for each training row(784 pixels of an image)-> tensor of probabilities is generated (here, of size [1x500]) (total 55000x500 vectors generated)

#sample the activation vector, from prab. distribution of hidden layer values
#take tensor of prob.(of sigmoidal activation) and make samples from all the distributions, h0


#forward pass-> data to hidden layer-> W*X + b -> relu activation on the result (difference between calculated W*X+b and the random hidden layer value
#these random value will be updated
X = tf.placeholder("float", [None, 784])
_h0 = tf.nn.sigmoid(tf.matmul(X, W) + hb) #probability of hidden units
#relu: activation func: -ve->0 +ve->1
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0)))) #sample_h_given_X



#example of relu and random_uniform
with tf.Session() as sess:
	a = tf.constant([0.7, 0.1, 0.8, 0.2])
	print(sess.run(a))
	b = sess.run(tf.random_uniform(tf.shape(a))) #generate uniform random varibale
	print(b)
	print(sess.run(a-b))
	print('tf.sign\n')
	print(sess.run(tf.sign(a-b))) #arithematic op with sign(+/-)
	print('relu\n')
	print(sess.run(tf.nn.relu(tf.sign(a-b)))) #activation func



#2. Backward pass (Reconstruction)
#input: samples from hidden layer (h0)
#same weight and bias-> go through sigmoid function
#produced output: reconstruction(approximation of original input)

_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb) #or tf.matmul(W, h0)
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))#sample_v_given_h
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

#reconstruction steps
#Pass 0: X->X:_ho(X converted to _h0/calculated) -> h0:v1 (h0 conv. to v1) (v1 is reconstruction of first pass
#Pass 1: v1-> v1:_h1->h1:v2 (v2 is reconstruction of second pass)


#maximise product of probabilities assigned to training set V

#calculate gradient(for gradient descent)
#define objective function as average negative log-likelihood & try to minimise it
#use stochastic gradient descent to find optimal weight and biases

#positive phase: increase probability of training data
#negative phase: decrease the probability of samples generated by model

#negative phase is hard to compute: so Contrastive Divergence (CD)
#gives correct direction of gradient estimate
#use "Gibbs sampling" to sample from our model distribution

#CD is matrix of values, used to adjust W matrix values
#on each step(epoch)
#W' = W + alpha*CD; alpha: learning rate

#calculate CD
#single step CD (CD-1)
	#take input sample, compute prob. of hidden unit(_h) and sample a hidden activation vector(h)
	#_h0 = sigmpid(W*X + hb)
	#h0 = sampleProb(h0)
	#positive gradient: X x h0:(call it w_pos_grad)(x:outer product)

#Gibbs sampling: from h, reconstruct v1->take smaple of visible unit, resample hidden activation h1 from this
	#_v1 = sigmoid(h0xW(transpose) + vb)
	#v1 = sampleProb(v1) (sample v given h)
	#h1 = sigmoid(v1xW + hb)

	#negative gradient (w_neg_grad): v1 x h1 (reconstruction 1)
 
#CD = (w_pos_grad - w_neg_grad)/datapoints
#W' = W + alpha*CD

#at end, visible nodes will store value of sample
#forward pass->randomly set values of each hi value to be 1, with prob. sigmoid(v*W + hb)
#reconstruction->randomly set values of each vi to be 1 with probability sigmoid(h*transpose(W) + vb)


#x outer_product y: tf.matmul(tf.transpose(x), y)
alpha = 1.0
w_pos_grad = tf.matmul(tf.transpose(X), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(X)[0])
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(X - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

#objective->maximise likelihood of data to be drawn from that distribution
#calculate error: square mean difference of input(X) and final reconstruction output(v1) (can check it for all step 1 to n)

err = tf.reduce_mean(tf.square(X - v1))

#start session and initialise

cur_w = np.zeros([784,500], np.float32) #current weight
cur_vb = np.zeros([784], np.float32) 	#current visible_bias
cur_hb = np.zeros([500], np.float32)	#current hidden_bias
prv_w = np.zeros([784,500], np.float32) #previous weight
prv_vb = np.zeros([784], np.float32)	#previous visible_bias
prv_hb = np.zeros([500], np.float32)	#previous hidden_bias

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#first run err
print('first error\n')
print(sess.run(err, feed_dict={X: trX, W: prv_w, vb: prv_vb, hb: prv_hb}))

#Parameters
epochs = 5
batchsize = 100
weights = []
errors = []

for epoch in range(epochs):
	for start, end in zip( range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
		batch = trX[start:end]
		cur_w = sess.run(update_w, feed_dict={X: batch, W: prv_w, vb: prv_vb, hb:prv_hb})
		cur_vb = sess.run(update_vb, feed_dict={X: batch, W: prv_w, vb: prv_vb, hb:prv_hb})
		cur_hb = sess.run(update_hb, feed_dict={X: batch, W: prv_w, vb: prv_vb, hb:prv_hb})
		prv_w = cur_w
		prv_vb = cur_vb
		prv_hb = cur_hb
		
		#record regular error stats
		if start % 20000 == 0:
			errors.append(sess.run(err, feed_dict={X: trX, W: cur_w, vb: cur_vb, hb:cur_hb}))
			weights.append(cur_w)
	print('Epoch %d' % epoch, 'reconstruction error#last step: %f' % errors[-1])
plt.plot(errors)
plt.xlabel('Batch Number')
plt.ylabel('Error')
plt.show()

#last weight after training
uw = weights[-1].T	#transpose
print('weight after last training\n', uw)
