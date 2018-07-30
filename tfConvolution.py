#tensorflow convolution

#tensorflow creates all the operations in form of graphs and execute them once with highly optimized backend

import tensorflow as tf

#Building graph

#3x3 filter (4D tensor: [3,3,1,1] ([width, height, channels, no. of filters]))
#10x10 image (4D tensor: [1,10,10,1] ([batch size, width, height, no. of channels))

input = tf.Variable(tf.random_normal([1,10,10,1]))
filter = tf.Variable(tf.random_normal([3,3,1,1]))

op = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='VALID')
op2 = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')

#initialize and run session
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	
	print('Input \n')
	print('{0} \n'.format(input.eval()))
	print("Filter/kernel \n")
	print('{0} \n'.format(filter.eval()))
	print("result/Feature Map with valid positions \n")
	result = sess.run(op)
	print(result)
	print('\n')
	print("result/Feature Map with padding_same \n")
	result2 = sess.run(op2)
	print(result2)
