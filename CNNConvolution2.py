#Covolution:2D operation with Python
#kernel: 1 dimension

#I' = Summation(I(x -u, y - v)g(u,v))

from scipy import signal as sg
I = [[255, 7, 3],[212, 240, 4],[218, 216, 230],]

g = [[-1,1]]

print("Without zero padding\n")
print('{0} \n'.format(sg.convolve( I, g, 'valid')))
#valid argument: output consist of only those element that dont rely on zero-padding

print("With zero padding\n")
print(sg.convolve(I, g)) #by default: with zero-padding

#if kernel: 2 dimiension
#h = [[-1,1],
#     [2 ,3]]

#then inverse of it will be like: 3	2
#				  1     -1

I = [[255, 7, 3], [212, 240, 4], [218, 216, 230],]

g = [[-1, 1], [2, 3]]

print('With zero padding \n')
print('{0} \n'.format(sg.convolve(I, g, 'full')))

print('zero padding_same\n')
print('{0} \n'.format(sg.convolve(I, g, 'same')))

print('Without zero padding \n')
print(sg.convolve( I, g, 'valid'))
