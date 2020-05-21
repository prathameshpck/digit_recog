from utility.dataset_loader import load
from utility.functions import *
from progressbar import ProgressBar
import numpy as np
from time import sleep
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold = sys.maxsize)
x_train,y_train, x_test,y_test = load()

factor = 0.99/255

x_train = ((x_train - np.mean(x_train))/ np.std(x_train))/255 

#x_train = (x_train * factor)+0.01


x_train = x_train.reshape(60000,784).T
# print(x_train[:,0])
# x_train = ((x_train - np.mean(x_train))/ np.std(x_train))

x_test = x_test.reshape(10000,784)

y_train = hotkey(y_train)
y_test = hotkey(y_test)


class network:
	def __init__(self,x,y):
		self.cache = {}
		self.cache['x'] = x[:,0].reshape(784,1)
		self.cache['y'] = y[0].reshape(10,1)
		self.cache['w1'] = np.random.random((28,784))
		self.cache['w2'] = np.random.random((16,28))
		self.cache['w3'] = np.random.random((28,16))
		self.cache['w4'] = np.random.random((10,28))
		self.cache['b1'] = np.random.random((28,1))
		self.cache['b2'] = np.random.random((16,1))
		self.cache['b3'] = np.random.random((28,1))
		self.cache['b4'] = np.random.random((10,1))
		
		


	def forward(self, x = None):
		if x is None:
			x = self.cache['x']
		# print()
		#print(x)
		w1,w2,w3,w4,b1,b2,b3,b4 = self.get('w1','w2','w3','w4','b1','b2','b3','b4')
		z1 = np.dot(w1,x) + b1
		a1 = sigmoid(z1)
		dz1 = sigmoid_prime(z1)
		

		z2 = np.dot(w2,a1) + b2
		a2 = sigmoid(z2)
		dz2 = sigmoid_prime(z2)

		z3 = np.dot(w3,a2) +b3
		a3 = sigmoid(z3)
		dz3 = sigmoid_prime(z3)

		z4 = np.dot(w4,a3) + b4
		a4 = softmax(z4)
		
		#print(a4, end = '\n')

		self.put(dz1=dz1,dz2=dz2,dz3=dz3,a4=a4,a3=a3,a2=a2,a1=a1)
		return a4

	def cost(self,y_hat,y = None):
		if y is None:
			y = self.cache['y'].T
		#print(y.shape,y_hat.shape)
		cost = -np.sum(np.sum(y.T*np.log(y_hat),axis = 0))

		return cost/1

	def backward(self):
		
		a4,a3,a2,a1,y,w4,w3,w2,w1,dz3,dz2,dz1,x = self.get('a4','a3','a2','a1','y','w4','w3','w2','w1','dz3','dz2','dz1','x')

		theta4 = (a4 - y)
		theta3 = np.dot(theta4.T,w4) * dz3.T
		theta2 = np.dot(theta3,w3) * dz2.T
		theta1 = np.dot(theta2,w2) * dz1.T

		dw4 = np.dot(theta4,a3.T)/1
		dw3 = np.dot(theta3.T,a2.T)/1
		dw2 = np.dot(theta2.T,a1.T)/1
		dw1 = np.dot(theta1.T,x.T)/1

		print(theta4.shape , w4.shape,dz3.shape,theta3.shape,a2.shape,dw3.shape)

		self.put(dw4 = dw4,dw2=dw2,dw3=dw3,dw1=dw1)

	def update(self,rate = 0.05):
		w1,w2,w3,w4,dw1,dw2,dw3,dw4 = self.get('w1','w2','w3','w4','dw1','dw2','dw3','dw4')

		w1 -= rate*dw1
		w2 -= rate*dw2
		w3 -= rate*dw3
		w4 -= rate*dw4

		# print(dw3)
		# print()

		self.put(w1=w1,w2=w2,w3=w3,w4=w4)


	def put(self,**kwargs):
		for key,value in kwargs.items():
			self.cache[key] = value

	def get(self,*args):
		u = tuple(map(lambda t: self.cache[t] , args))
		return u


def main():
	bar = ProgressBar()
	costs = []
	net = network(x_train,y_train)
	#print(net.forward())
	for i in (range(100)):
		out = net.forward()
		# if i%100 == 0 and i!=0:
		costs.append(net.cost(out))
		net.backward()
		net.update()	

	print(np.argmax(net.forward(), axis = 0))
	print(net.cache['y'])

	# 	print(costs)

	# plt.plot(costs)

	# plt.show()
	# plt.close()



if __name__ == '__main__':
	main()
