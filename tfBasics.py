import tensorflow as tf

#we define constants(edges) then define operation for those constants
#all computation in tensorflow happens by building graphs
#graphs are built by making operations and then running them using a session

hello = tf.constant('Hello World!')
print(hello)

a = tf.constant([2])
b = tf.constant([3])
c = tf.add(a,b)

#session create
session = tf.Session()
#run the session to use operation, specify operation to run
result = session.run(c)

print(result)
print(hello) #prints data type info

hello = session.run(hello)
print(hello) #print the content of hello

#session close
session.close()

