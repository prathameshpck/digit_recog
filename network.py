from utility.dataset_loader import load
from utility.functions import *
from progressbar import ProgressBar
import numpy as np
from time import sleep
import sys
import matplotlib.pyplot as plt


bar = ProgressBar()
np.set_printoptions(threshold = sys.maxsize)
x_train,y_train, x_test,y_test = load()

factor = 0.99/255

x_train = ((x_train - np.mean(x_train))/ np.std(x_train))/255 

#x_train = (x_train * factor)+0.01
print(y_train[0:20])

x_train = x_train.reshape(60000,784).T
# print(x_train[:,0])
# x_train = ((x_train - np.mean(x_train))/ np.std(x_train))

x_test = x_test.reshape(10000,784).T

y_train = hotkey(y_train)
y_test = hotkey(y_test)


class network:
	def __init__(self,x,y):
		self.cache = {}
		self.cache['x'] = x#[:,0:10]#[:,0].reshape(784,1)
		self.cache['y'] = y#[0:10].T #[0].reshape(10,1)
		self.cache['w1'] = np.random.random((784,28))
		self.cache['w2'] = np.random.random((28,16))
		self.cache['w3'] = np.random.random((16,28))
		self.cache['w4'] = np.random.random((28,10))
		self.cache['b1'] = np.random.random((28,1)).T
		self.cache['b2'] = np.random.random((16,1)).T
		self.cache['b3'] = np.random.random((28,1)).T
		self.cache['b4'] = np.random.random((10,1)).T
		
		


	def forward(self, x = None):
		if x is None:
			x = self.cache['x']
		# print()
		# print(x[:,0] - x[:,1])
		# print(x[:,1])
		#print(x)
		w1,w2,w3,w4,b1,b2,b3,b4 = self.get('w1','w2','w3','w4','b1','b2','b3','b4')
		#print(np.dot(w1.T,x))
		z1 = np.dot(w1.T,x ) #+ b1
		# print(np.dot(w1.T,x))

		a1 = relu(z1)
		dz1 = relu_prime(z1)

		#print(a1)
 
		z2 = np.dot(w2.T,a1) #+ b2
		# print(a1)

		a2 = relu(z2)
		dz2 = relu_prime(z2)



		z3 = np.dot(w3.T ,a2) #+b3
		a3 = relu(z3)
		dz3 = relu_prime(z3)

		z4 = np.dot(w4.T,a3) #+ b4
		a4 = softmax(z4)

		#print(z1,z2,z3,z4 , sep='\n')
		
		#print(a4.T, end = '\n')

		self.put(dz1=dz1,dz2=dz2,dz3=dz3,a4=a4,a3=a3,a2=a2,a1=a1)
		return a4

	def cost(self,y_hat,y = None):
		if y is None:
			y = self.cache['y'].T
		#print(y.shape,y_hat.shape)
		cost = -np.sum(np.sum(y.T*np.log(y_hat),axis = 0))

		return cost/32

	def train(self,epoch = 100):
		x = self.cache['x']
		y = self.cache['y']
		costs = []
		batches = np.split(x,1875,axis=1)
		targets = np.split(y,1875)
		for i in bar(range(epoch)):
			for batch,target in zip(batches,targets):
				out = self.forward(x = batch)
				costs.append(self.cost(out,target))
				self.backward(y=target,x=batch)
				self.update()

		return costs



	def backward(self,y=None,x=None):
		if y is None:
			y = self.cache['y']

		if x is None:
			x = self.cache['x']

		a4,a3,a2,a1,w4,w3,w2,w1,dz3,dz2,dz1= self.get('a4','a3','a2','a1','w4','w3','w2','w1','dz3','dz2','dz1')

		theta4 = (a4 - y.T)
		theta3 = np.multiply(dz3,np.dot(w4,theta4))
		theta2 = np.multiply(dz2,np.dot(w3,theta3))
		theta1 = np.multiply(dz1,np.dot(w2,theta2))

		dw4 = np.dot(theta4,a3.T)/32
		dw3 = np.dot(theta3,a2.T)/32
		dw2 = np.dot(theta2,a1.T)/32
		dw1 = np.dot(theta1,x.T)/32

		#print(theta4.shape , w4.shape,dz3.shape,theta3.shape,a2.shape,dw3.shape)

		self.put(dw4 = dw4,dw2=dw2,dw3=dw3,dw1=dw1)

	def update(self,rate = 0.005):
		w1,w2,w3,w4,dw1,dw2,dw3,dw4 = self.get('w1','w2','w3','w4','dw1','dw2','dw3','dw4')

		w1 -= rate*dw1.T
		w2 -= rate*dw2.T
		w3 -= rate*dw3.T
		w4 -= rate*dw4.T

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
	net = network(x_train,y_train)

	# for i in (range(1000)):
	# 	out = net.forward()
	# 	# if i%10 == 0 and i!=0:
	# 	costs.append(net.cost(out))
	# 	net.backward()
	# 	net.update()	

	costs = net.train(epoch = 100)

	costs = [cost for i,cost in enumerate(costs) if i%1000 == 0]

	# print(np.argmax((net.forward(net.cache['x'][:,0:50]).T), axis = 1) )
	# print(np.argmax(net.cache['y'][0:50] , axis = 1).T)	
	pred = np.argmax((net.forward(x_test).T), axis = 1).reshape(10000,1)
	y = np.argmax(y_test , axis = 1).T.reshape(10000,1)

	print(accuracy(pred,y))

		



	#print(costs)

	plt.plot(costs , markevery = 1000)

	plt.show()

	# plt.show(block=False)
	# #plt.pause(6)
	# #plt.close()



if __name__ == '__main__':
	main()
