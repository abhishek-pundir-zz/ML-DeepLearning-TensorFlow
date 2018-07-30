import tensorflow as tf

#variables are defined by tf.Variable()
#for using variables, first its necessary to intialise them, tf.global_variables_initializer()

#to update the value, run an assign operation tf.assign(data_member, new_value)

#increment a state (int type)
state = tf.Variable(0)
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

#variable must be initialised by running intialisation operation

init_op = tf.global_variables_initializer()

with tf.Session() as session:
	#run initializer
	session.run(init_op)
	print(session.run(state))
	
	for _ in range(3):
		session.run(update)
		print(session.run(state))

session.close()
		

