#LSTM basics

import numpy as np
import tensorflow as tf
sess = tf.Session()

#create network of single LSTM cell
#pass 2 input to cell: prev_output, prev_state (h and c)
#initialize a state vector "state"->tuple of 2 elements, each one of size [1x4]: one for passing prv_output to next time step & one for passing prv_sate

LSTM_CELL_SIZE = 4 #output size (dimension) same as hidden size in cell

lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True) #creat the lstm_cell argument: cell size,cell is tuple

state = (tf.zeros([2,LSTM_CELL_SIZE]),)*2 #state tensor created


#sample input: batch size=2, seq_len=6
sample_input = tf.constant([[1,2,3,4,3,2],[3,2,2,2,2,2]], dtype=tf.float32)

print(sess.run(sample_input))

#pass the input to lstm_cell

with tf.variable_scope("LSTM_sample1"):
	output, state_new = lstm_cell(sample_input, state)
sess.run(tf.global_variables_initializer())

print(sess.run(state_new)) #c:state, h:output

print(sess.run(output)) #NEW DATA


