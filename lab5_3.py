#LSTM on MNIST
import warnings

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#builtin mnist dataset in tensorflow
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True)
#.read_data_sets() loads entire data set and returns an object

#mnist dataset has train|test image|label
trainimgs = mnist.train.images
trainlabels = mnist.train.labels
testimgs = mnist.test.images
testlabels = mnist.test.labels

ntrain = trainimgs.shape[0]
ntest = testimgs.shape[0]
dim = trainimgs.shape[1]
nclasses = trainlabels.shape[1]

print('Train images: ', trainimgs.shape)
print('Train labels: ', trainlabels.shape)
print('\nTest Images: ', testimgs.shape)
print('\nTest labels: ',testlabels.shape)

#one sample of mnist
samplesIdx = [100,101,102]

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

ax1 = fig.add_subplot(121)
ax1.imshow(testimgs[samplesIdx[0]].reshape([28,28]), cmap='gray')

xx, yy = np.meshgrid(np.linspace(0,28,28), np.linspace(0,28,28))
X = xx; Y = yy
Z = 100*np.ones(X.shape)

img = testimgs[77].reshape([28,28])
ax = fig.add_subplot(122, projection='3d')
ax.set_zlim((0,200))

offset=200
for i in samplesIdx:
	img = testimgs[i].reshape([28,28]).transpose()
	ax.contourf(X, Y, img, 200, zdir='z', offset=offset, cmap="gray")
	offset -= 100
	ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

plt.show()

for i in samplesIdx:
	print("sample: {0} - Class: {1} - Label Vector: {2}".format(i, np.nonzero(testlabels[i])[0], testlabels[i]))


#sample data contain 28x28 dimension input
#RNN contains: input layer->converting 28x28 to 128 dimension hidden layer
#one intermediate Recurrent neural network (RNN)
#one output layer which converts 128 dim output of LSTM to 10 dim o/p indicating a class label (0 to 9)

n_input = 28 #MNIST data i/p shape
n_steps = 28 #timespaces
n_hidden = 128 #hidden layer num of features
n_classes = 10 #MNIST total classes (0-9 digits)


learning_rate = 0.001
training_iters = 100000
batch_size = 100
display_step = 10

#construct RNN
#tensor shape: [batch_size, time_steps,input_dimension] (here (?,28,28)
x = tf.placeholder(dtype="float", shape=[None, n_steps, n_input], name="x") #current data->batch_size:100
y = tf.placeholder(dtype="float", shape=[None, n_classes], name="y")

#create weights and biases
#weight matrix: n_hiddenxn_classes (10x10)
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}
#create lstm_cell
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

#dynamic_rnn create rnn specified from lstm_cell
outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs=x, dtype=tf.float32)

output = tf.reshape(tf.split(outputs, 28, axis=1, num=None, name='split')[-1],[-1,128])
#pred contains the calculated class
pred = tf.matmul(output, weights['out']) + biases['out']

print(pred)

#define cost function and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#define accuracy and evaluation methods

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#initialise variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	step = 1
	#keep training untill max iteration reached
	while step * batch_size < training_iters:
		#read batch of 100 images [100x784] (28x28=784)image pixels as batch_x
		#batch_y is a matrix of [100x10] 100 images k 10 class labels
		batch_x, batch_y = mnist.train.next_batch(batch_size)

		#consider each row of  image as separate sequence
		#reshape data to get 28 seq of 28 elements so that, batch_x:[100x28x28]
		batch_x = batch_x.reshape((batch_size, n_steps, n_input))
		#run optimisation
		sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})
		#check accuracy at arbitrary steps
		if step%display_step == 0:
			#calculate batch accuracy
			acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
			#calculate batch loss
			loss = sess.run(cost, feed_dict={x:batch_x, y:batch_y})
			print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
		step += 1

	print("Optimisation Finished!'")
	#Calculate accuracy for 128 mnist test images
	test_len = 128
	test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
	test_label = mnist.test.labels[:test_len]
	print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y:test_label}))
	sess.close()
