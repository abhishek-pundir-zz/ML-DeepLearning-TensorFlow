import tensorflow as tf

#placeholders are used to pass values at runtime
#just define the function wid placeholder, pass value during runtime

#to pass value at runtime, value is passed as dictionary
#{placeholder_name: value}
#feed_dict is the dictionary passed as argument

#define edges (constant)

a = tf.placeholder(tf.float32)

b = a*2

with tf.Session() as session:
	result = session.run(b, feed_dict={a:3.5})
	print(result)

session.close()
