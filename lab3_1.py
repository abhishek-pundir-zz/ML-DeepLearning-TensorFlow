#activation function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D	#available with matplot package

#plot for arbitrary activation function, for weights and biases in range -0.5 to 0.5, increment: 0.05
#def plot_act(i=1.0, actfunc=lambda x: x):
#	#weights: ws & biases: bs
#	ws = np.arange(-0.5, 0.5, 0.05)
#	bs = np.arange(-0.5, 0.5, 0.05)

#	X, Y = np.meshgrid(ws, bs)
	
#	os = np.array([actfunc(tf.constant(w*i + b).eval(session=sess) for w,b in zip(np.ravel(X), np.ravel(Y)))])
	
#	Z = os.reshape(X.shape)

#	fig = plt.figure()
#	ax = fig.add_subplot(111, projection='3d')
#	ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
#

def plot_act(i=1.0, actfunc=lambda x: x):
    ws = np.arange(-0.5, 0.5, 0.05)
    bs = np.arange(-0.5, 0.5, 0.05)

    X, Y = np.meshgrid(ws, bs)

    os = np.array([actfunc(tf.constant(w*i + b)).eval(session=sess) \
                   for w,b in zip(np.ravel(X), np.ravel(Y))])

    Z = os.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
    plt.show()

#basic structure
#weighted sum go into neuron, produce some output

sess = tf.Session()
#input of 3 real values; shape:[1x3]
i = tf.constant([1.0, 2.0, 3.0], shape=[1, 3])
#matrix of weights:[3x3]
w = tf.random_normal(shape=[3, 3])
#vector of bias: [1x3]
b = tf.random_normal(shape=[1, 3])
#dummy activation function
def func(x): return x
#perform i*w + b
#tf.matmul: matrix multiplication
act = func(tf.matmul(i, w) + b)
#Evaluate tensor to numpy array: eval()
act.eval(session=sess)
print('activation function\n')
print(act)

#put in activation function, plot_act
plot_act(1.0, func)


#sigmoid function
plot_act(1, tf.sigmoid)
#sigmoid in neural network
act = tf.sigmoid(tf.matmul(i, w)+b)
act.eval(session=sess)
print('\nsigmoid activation\n')
print(act)

#TanH
plot_act(1, tf.tanh)
#using TanH in neural network
act = tf.tanh(tf.matmul(i, w) + b)
act.eval(session=sess)
print('\ntanh activation\n')
print(act)

#Linear Unit function: based on LeRU
plot_act(1, tf.nn.relu)
