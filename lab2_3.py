#logistic regression
#it passes the value through logistic/sigmoid curve but treats them as probability

#logistic regression is a probabilistic classification
#theta(y) = e^y/(1+e^y)

import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

#using the iris dataset, this dataset is inbuilt

iris = load_iris()
iris_X, iris_y = iris.data[:-1,:], iris.target[:-1]
iris_y = pd.get_dummies(iris_y).values #CATEGORY VALUES

trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)

#define x and y placeholders, to hold values of data and label, placeholder can accept any number of data inputs, it just knows the shape of the input data

#numFeatures - no. of feature (attributes) in our input, in iris dataset->4 features(sepal|petal length|width)
numFeatures = trainX.shape[1]

#numLabels - no. of classes (classes of probabilistic classification) of input, iris-> 3 classes
numLabels = trainY.shape[1]

#placeholders
#'None' means TensorFlow shouldnt expect a fixed number here
X = tf.placeholder(tf.float32, [None, numFeatures]) #iris has 4 features so, X will hold an input set of 4 attr. each
yGold = tf.placeholder(tf.float32, [None, numLabels]) #this will be correct answer matrix of 3 labels

#Set model weights and bias
#Y = X*W + b -> W is weight, and b is bias, initial value of both is all zero
#both will be variables
#W shape: [4x3] since data:[nx4] final result[nx3], b shape[1x3] (or [3]) to add to final value
# 3-d output: [0,0,1],[0,1,0],[1,0,0]
W = tf.Variable(tf.zeros([4,3])) #4-d input and 3 classes
b = tf.Variable(tf.zeros([3])) #3-d output

#set initial value of W and b: random,with standar deviation of 0.1

weights = tf.Variable(tf.random_normal([numFeatures,numLabels],mean=0,stddev=0.01,name="weights"))
bias = tf.Variable(tf.random_normal([1,numLabels],mean=0,stddev=0.1,name="bias"))

#Logistic Regression Model  y' = sigmoid(WX + b)
#1. Weight x features matrix operation
#2. summation of "weighted" features and bias
#3. applcation of sigmoid
#these 3 operation feed into each other, one after another(shown below)

#sigmoid function: tf.nn.sigmoid()
apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

#training
#goal of algo: find best weight
#more cost, more bad model
#cost function in our model: Square mean error loss function

#least square linear regression can't be used here->gradient descent use

#how long we are going to train->Epochs
numEpochs = 700

#learning rate
#learning rate iterations (decay); in logistic regression, learning rate decays over iterations
learningRate = tf.train.exponential_decay(learning_rate=0.0008,global_step=1,decay_steps=trainX.shape[0],decay_rate=0.95,staircase=True)

#Cost function: Squared mean error
cost_OP = tf.nn.l2_loss(activation_OP - yGold,name="squared_error_cost")
#Gradient descent (mention <learning rate>.<cost_function> to minimise
training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)

#create session and initialize variables

sess = tf.Session()
#Initialize weights and bias variables
init_OP = tf.global_variables_initializer()
sess.run(init_OP)

#additional operation to keep track of model's efficiency
#argmax(x,1) -> returns maximum value in the x column(data structure)
#1. argmax(activation_OP,1) - label with most probability
#2. argmax(yGold,1) is the correct label

#correct prediction where actual and calc. both value==1
correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1),tf.argmax(yGold,1))

###if every false=0 & every true=1 then, average returns accuracy
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float")) #typecasting

#Summaries
#summary op for regression output
activation_summary_OP = tf.summary.histogram("output", activation_OP)
#Summary op for accuracy
accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)
#Summary op for cost
cost_summary_OP = tf.summary.scalar("cost", cost_OP)
#Summary ops to check how vriables (W, b) are updating after each iteration
weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))
biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))

#Merge all summaries
merged = tf.summary.merge([activation_summary_OP, accuracy_summary_OP, cost_summary_OP, weightSummary, biasSummary])

#Summary writer
writer = tf.summary.FileWriter("summary_logs", sess.graph)


#training
cost = 0
diff = 1
epoch_values=[]
accuracy_values=[]
cost_values=[]

#training epochs
for i in range(numEpochs):
	if i > 1 and diff < .0001:
		print("change in cost %g; convergence."%diff)
		break
	else:
		#Run training step i.e sigmoid calculation
		step = sess.run(training_OP, feed_dict={X: trainX, yGold: trainY})
		#Report occasional stats of sigmoid calculation
		if i % 10 == 0:
			epoch_values.append(i)
			#Generate accuracy stats on this test data
			train_accuracy, newCost = sess.run([accuracy_OP, cost_OP], feed_dict={X: trainX, yGold: trainY})
			
			#Add accuracy to live graphing variable
			accuracy_values.append(train_accuracy)
			
			#Add cost to live graphing variable
			cost_values.append(newCost)
			
			#Re-assign values for variables
			diff = abs(newCost - cost)
			cost = newCost

			#generate print statements
			print("step %d, training accuracy %g, cost %g, change in cost %g" %(i,train_accuracy, newCost, diff))

#final accuracy of model on test set
print("Final accuracy on test set: %s" %str(sess.run(accuracy_OP, feed_dict={X: testX, yGold: testY})))


#plotting the cost (the last 50 iterations of occasional stats)
plt.plot([np.mean(cost_values[i-50:i]) for i in range(len(cost_values))])
plt.show()
