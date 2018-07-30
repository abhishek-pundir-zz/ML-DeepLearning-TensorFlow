#Convolution applied on images
#Edge Detection

#download standard image
#wget --quiet https://ibm.box.com/shared/static/cn7yt7z10j8rx6um1v9seagpgmzzxnlz.jpg --output-document bird.jpg

#already downloaded

#1. load image (.jpg format)
#2. convert image to grayscale
#3. convolve image with edge detector kernel, store result in new matrix; display matrix in image format
#4. normalize the matrix; update pixel values
#5. store it as new matrix; display the result

#importing
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image


#default file name: bird.jpg (enter in command line)
print("Enter name of the file (default: bird.jpg")
raw = raw_input()
im = Image.open(raw)

#converting image to greyscale (using Luma transform)
image_gr = im.convert("L")
print("\nOriginal type: %r \n\n" % image_gr)

###convert image to a matrix with values from 0 to 255 (uint8)
arr = np.asarray(image_gr)

print("After conversion to numerical representation: \n\n %r" % arr)

#activate matplotlib for ipython
#%matplotlib inline
#not needed

#plot image (using matplotlib.pyplot)

imgplot = plt.imshow(arr)
imgplot.set_cmap('gray') #other maps: gray, winter, autumn

print("\n Input image converted to gray scale: \n")
plt.show(imgplot)


#edge detector kernel
kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])

#gradient (new) after edge detector kernel slide over arr
grad = signal.convolve2d(arr, kernel, mode='same', boundary='symm')

print('GRADIENT MAGNITUDE - Feature Map')
fig, aux = plt.subplots(figsize=(10, 10))
#display image from the matrix
edgeplot = aux.imshow(np.absolute(grad), cmap='gray') #saving op in a variable to display
print("\n Grayscale image converted to edge detection \n")
plt.show(edgeplot)


#when dealing with real applications, convert pixel values to range from 0 to 1 -> called "normalization"

type(grad)

#new matrix grad_biases with updated pixel values
grad_biases = np.absolute(grad) + 100 #adding 100 to each pixel value
grad_biases[grad_biases > 255] = 255 #maximizing upto 255

print('GRADIENT_BIASES MAGNITUDE - Feature Map')
fig, aux = plt.subplots(figsize=(10, 10))
grad_biases_edgeplot = aux.imshow(np.absolute(grad_biases), cmap='gray')
plt.show(grad_biases_edgeplot)
