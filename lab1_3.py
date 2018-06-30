import tensorflow as tf

#creating placeholder of string passing value 'abhishek' to it

a = tf.placeholder(tf.string)
b = 'Hello ' + a
with tf.Session() as sess:
	result = sess.run(b,feed_dict={a:'abhishek'})
	print(result)
