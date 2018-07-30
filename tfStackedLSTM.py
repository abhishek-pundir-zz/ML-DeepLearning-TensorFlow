#Stacked LSTM (eg. 2 layer LSTM)

import numpy as np
import tensorflow as tf
sess = tf.Session()

LSTM_CELL_SIZE = 4 #4 hidden nodes, state_dim = output_dim
input_dim = 6
num_layers = 2

#create stacked lstm_cell

cells=[]
for _ in range(num_layers):
	#create lstm_cell (here 2 lstm_cell will get created
	cell = tf.contrib.rnn.LSTMCell(LSTM_CELL_SIZE)
	#add them to cells array	
	cells.append(cell)
stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells) #created multiRNN cell

#Now, create the RNN
#batch_size x time_steps x features(input_dim)

data = tf.placeholder(tf.float32, [None, None, input_dim])
#create rnn function: tf.nn.dynamic_rnn(cell,data,data_type)
output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

#eg. batch_size:2, input sequence length:3, features:6
#therefore, tensor shape(batch_size, time_step,dimension)=(2,3,6)

sample_input = [[[1,2,3,4,3,2], [1,2,1,1,1,2], [1,2,2,2,2,2]],[[1,2,3,4,3,2],[3,2,2,1,1,2],[0,0,0,0,3,2]]]

print(sample_input)

#send input to network
sess.run(tf.global_variables_initializer())
final_output = sess.run(output, feed_dict={data:sample_input})
print('final_output\n')
print(final_output)

final_state = sess.run(state, feed_dict={data:sample_input})
print('\nfinal_state\n')
print(final_state)
