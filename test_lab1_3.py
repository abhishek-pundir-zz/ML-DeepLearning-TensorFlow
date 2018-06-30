import tensorflow as tf

a = tf.placeholder(tf.string)
b = input('What is your name ')
result = 'Hello'+b

with tf.Session() as sess:
	sess.run(result)
	print(result)
