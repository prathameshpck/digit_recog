from utility.dataset_loader import load
from utility.functions import *
from progressbar import ProgressBar
import numpy as np
import itertools
import matplotlib.pyplot as plt

x_train,y_train, x_test,y_test = load()

factor = 0.99/255

# x_train = ((x_train - np.mean(x_train))/ np.std(x_train))

x_train = (x_train * factor)+0.01


x_train = x_train.reshape(60000,784)

# x_train = ((x_train - np.mean(x_train))/ np.std(x_train))

x_test = x_test.reshape(10000,784)

y_train = hotkey(y_train)
y_test = hotkey(y_test)


class network:
	def __init__(self,x,y):
		self.cache = {}
		self.cache['x'] = x.T
		self.cache['y'] = y
		self.cache['w1'] = np.random.random((28,x.shape[1]))
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

		w1,w2,w3,w4,b1,b2,b3,b4 = self.get('w1','w2','w3','w4','b1','b2','b3','b4')
		z1 = np.dot(w1,x) + b1
		a1 = relu(z1)
		dz1 = relu_prime(z1)

		z2 = np.dot(w2,a1) + b2
		a2 = relu(z2)
		dz2 = relu_prime(z2)

		z3 = np.dot(w3,a2) +b3
		a3 = relu(z3)
		dz3 = relu_prime(z3)

		z4 = np.dot(w4,a3) + b4
		a4 = softmax(z4)

		self.put(dz1=dz1,dz2=dz2,dz3=dz3,a4=a4,a3=a3,a2=a2,a1=a1)
		return a4

	def cost(self,y_hat,y = None):
		if y is None:
			y = self.cache['y'].T
		cost = -np.sum(np.sum(y*np.log(y_hat),axis = 0))

		return cost/60000

	def backward(self):
		pass


	def update(self,rate = 5):
		w1,w2,w3,w4,dw1,dw2,dw3,dw4 = self.get('w1','w2','w3','w4','dw1','dw2','dw3','dw4')

		w1 -= rate*dw1
		w2 -= rate*dw2
		w3 -= rate*dw3
		w4 -= rate*dw4

		self.put(w1=w1,w2=w2,w3=w3,w4=w4)


	def put(self,**kwargs):
		for key,value in kwargs.items():
			self.cache[key] = value

	def get(self,*args):
		u = tuple(map(lambda t: self.cache[t] , args))
		return u


def main():
	net = network(x_train,y_train)
	print(net.cache['y'].shape)
	out = net.forward()
	print(net.cost(out))

if __name__ == '__main__':
	main()
