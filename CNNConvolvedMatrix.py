#convolutionalNeuralNetwork
#uses neurons of same type
#1. sliding window (called kernel or filter)
#2. matrix of the input image
#Feature engineering: idetifing smallest feature and then combining them up and up to form full image

#"convolved" matrix is formed by sliding 'inverse' of kernel over the input matrix and summing them all

import numpy as np

h = [2,1,0] #image
x = [3,4,5] #kernel

y = np.convolve(x, h)
print(y)  #o/p: 6 11 14 5 0
#reverse x -> [5,4,3]
#slide x over h:
	#3*2 = 6
	#3*1 + 4*2 = 11
	#3*0 + 4*1 + 5*2 = 14
	#4*0 + 5*1 = 5
	#5*0 = 0

#equation of this operation: y[n] = SUMMATION(x[k].h[n-k]); k->0,1,2...

print("compare with following values: y[0] = {0} ; y[1] = {1} ; y[2] = {2} ; y[3] = {3} ; y[4] = {4}".format(y[0],y[1],y[2],y[3],y[4]))

#Method to apply kernel on matrix
	#1. Operation with Padding (full): insert a zero on left and right of input matrix (each row) 
	#2. Operation Padding (Same): insert a zero on left only (of each row); this returns same number of elements in convolved matrix as in input image matrix
	#3. Without Padding (Valid): no zero inserted; no. of elements = input dimension - kernel dimension + 1
#eg: input image: 10x10; kernel:3x3 therefore, Valid operation elements (in each row) = 10 - 3 + 1 = 8


#Full
x = [6,2]
h = [1,2,5,4]

y = np.convolve(x, h, "full")
print("full operation\n")
print(y)

#Same
y = np.convolve(x, h, "same")
print("same operation")
print(y)

#Valid
y = np.convolve(x, h, "valid")
print("valid operation")
print(y)
