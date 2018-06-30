#Autoencoders
#using MNIST dataset

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10	#no. of images to be displayed

#Network Parameters
n_hidden_1 = 256 #1st layer number of features
n_hidden_2 = 128 #2nd layer number of features
n_input = 784 #MNIST data input (28 x 28)


#tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

#weights for encoder: input-h1, h1-h2
#weights for decoder: h2-h1, h1-input
weights = {
	'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
	'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}

#biases for 2 encoders and 2 decoders (input-h1, h1-h2)
biases = {
	'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}




#encoder (2 layer)
#building the encoder
#input: X and ouput: layer_2

def encoder(x):
	#encoder 1st layer with sigmoid activation
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
	
	#encoder 2nd layer with sigmoid activation (layer_1 x weight['encoder_h2'])
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))

	return layer_2

#decoder
#reverse of encoder
#layer_1 of encoder will be layer_2 of decoder
#(input will be return of encoder (layer_2) and o/p: reconstructed input)

def decoder(x):
	#Decoder first layer with sigmoid activation
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))

	#decoder 2nd layer with sigmoid activation
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))

	return layer_2



###Construct Model
#cost -> loss function (tf.reduce_mean())
#optimizer -> gradient (tf.train.RMSPropOptimizer) (for backpropagation)

encoder_op = encoder(X) #value of encoded input in encoder_op (dimensionality reduced)
decoder_op = decoder(encoder_op) #reconstructed input in decoder_op, after slecting important features and dimensionality reduction

#Prediction
y_pred = decoder_op

#Targets (Labels) are the input data
y_true = X

###Define Loss and optimizer, minimize the squared error
#cost: RMS of y_true and y_pred
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

#optmize with learning_rate, minimizing the cost
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


#Initializing varibales
init = tf.global_variables_initializer()



###running training for 20 epochs

#launch the graph (using InteractiveSession (it is for notebooks))
sess = tf.InteractiveSession()
sess.run(init)

total_batch = int(mnist.train.num_examples/batch_size) #total number of batches

#Training cycle (training_epochs = 20)
for epoch in range(training_epochs):
	#Loop over all batches
	for i in range(total_batch):
		batch_xs, bathc_ys = mnist.train.next_batch(batch_size)
		#run optimization op(backpropagation) and cost operation(to get loss value)
		_, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
	
	#Display logs per epoch step
	if epoch % display_step == 0:
		print("Epoch: ", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

print("Optimisation finished")



###Apply encode decode over test set
encode_decode = sess.run(y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})


###Visualise results
#compare original images with their reconstruction
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
	a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
	a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))

