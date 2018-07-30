#linear regression
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

#generate random points and define linear relation

x_data = np.random.rand(100).astype(np.float32)

#equation for this model : Y=3X+2	desired model

y_data = 3*x_data + 2

#adding some gaussian noise to the data of y_data
y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.1))(y_data)

#initialize a & b with random value
a = tf.Variable(1.0) #slope
b = tf.Variable(0.2) #intercept
#define the linear function
y = a * x_data + b

#minimise the squared error, define equation to be minimized as loss
#find loss value using tf.reduce_mean()

loss = tf.reduce_mean(tf.square(y - y_data)) #mean of square diff of real and ideal

#optimizer method
#gradient descent with learning rate 0.5
#minimise the loss, tf.train.GradientDescentOptimizer.minimize()

optimizer = tf.train.GradientDescentOptimizer(0.5) #learning rate specified

train = optimizer.minimize(loss)

#initialise the variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#display template
print("(Step, [slope, intercept])")
#start optimization and run the graph
train_data = []
for step in range(100):
	evals = sess.run([train,a,b])[1:] #train is object of loss minimizer function, it is passed with a and b values
	if step % 5 == 0:
		print(step, evals)
		train_data.append(evals)

#graph plotting

converter = plt.colors
cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
	cb += 1.0 / len(train_data) #to produce color spectrum
	cg -= 1.0 / len(train_data)
	if cb > 1.0: cb = 1.0
	if cg < 0.0: cg = 0.0
	[a, b] = f #f is train data value [1x2 type data for a&b]
	f_y = np.vectorize(lambda x: a*x + b)(x_data)
	line = plt.plot(x_data, f_y)
	plt.setp(line, color=(cr, cg, cb))

plt.plot(x_data, y_data, 'ro')

green_line = mpatches.Patch(color='red', label='Data Points') #for legend in graph

plt.legend(handles=[green_line])

plt.show()

#in the graph one ideal line is visible, in cluster of lines, initially the slope of line was very high, then gradient descent optimizer minimised the error and final line is very parallel to ideal line
