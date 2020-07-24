from utility.dataset_loader import load
from utility.functions import *
from progressbar import ProgressBar
import numpy as np
from time import sleep
import sys
import itertools
import matplotlib.pyplot as plt
import csv

bar = ProgressBar()

x_train,y_train, x_test,y_test = load()


x_train = ((x_train - np.mean(x_train))/ np.std(x_train))/255 
x_train = x_train.reshape(60000,784).T

x_test = x_test.reshape(10000,784).T

devx = x_test[: , 9000:]
devy = y_test[9000:] 

x_test = x_test[: , :9000].T
y_test = y_test[:9000]

y_train = hotkey(y_train)
y_test = hotkey(y_test)

l = 0.00001

class network:
	def __init__(self,x,y):
		self.cache = {}
		self.cache['x'] = x
		self.cache['y'] = y
		self.cache['w1'] = np.random.random((784,310)) * np.sqrt(2/784)
		self.cache['w2'] = np.random.random((310,270)) * np.sqrt(2/310)
		self.cache['w3'] = np.random.random((270,250)) * np.sqrt(2/270)
		self.cache['w4'] = np.random.random((250,10)) * np.sqrt(2/250)
		self.cache['b1'] = np.random.random((310,1))
		self.cache['b2'] = np.random.random((270,1))
		self.cache['b3'] = np.random.random((250,1))
		self.cache['b4'] = np.random.random((10,1))
		self.cache['vdw'] = np.array([np.zeros((784 ,310)) , np.zeros((310,270)) , np.zeros((270,250)) , np.zeros((250,10)) ] , dtype = object)
		self.cache['sdw'] = [np.zeros((784 , 310)) , np.zeros((310,270)) , np.zeros((270,250)) , np.zeros((250,10)) ]
		self.cache['vdb'] = [np.zeros((310 , 1)) , np.zeros((270,1)) , np.zeros((250,1)) , np.zeros((10,1)) ]
		self.cache['sdb'] = [np.zeros((310 , 1)) , np.zeros((270,1)) , np.zeros((250,1)) , np.zeros((10,1)) ]

		with open("accuracy.csv" , "w") as f:
			t = csv.DictWriter(f , fieldnames = ['cost' , "accuracy",'accuracy_dev'])
			t.writeheader()

		


	def forward(self, x = None):
		if x is None:
			x = self.cache['x']
		w1,w2,w3,w4,b1,b2,b3,b4 = self.get('w1','w2','w3','w4','b1','b2','b3','b4')
		
		z1 = np.dot(w1.T,x) + b1
		a1 = relu(z1)
		dz1 = relu_prime(z1)

		z2 = np.dot(w2.T,a1) + b2
		a2 = relu(z2)
		dz2 = relu_prime(z2)

		z3 = np.dot(w3.T,a2) + b3
		a3 = relu(z3)
		dz3 = relu_prime(z3)

		z4 = np.dot(w4.T,a3) + b4
		a4 = softmax(z4)


		self.put(dz1=dz1,dz2=dz2,dz3=dz3,a4=a4,a3=a3,a2=a2,a1=a1)
		return a4

	def cost(self,y_hat,y = None):
		if y is None:
			y = self.cache['y'].T

		w1,w2,w3,w4 = self.get('w1','w2','w3','w4')

		cost = -np.sum(np.sum(y.T*np.log(y_hat),axis = 0))/32

		reg = (l/2*32) *(np.sum(np.square(w1)) + np.sum(np.square(w2)) + np.sum(np.square(w3))+ np.sum(np.square(w4)))
	
		return cost + reg
		

	def train(self,epoch = 100):
		x = self.cache['x']
		y = self.cache['y']
		costs = []
		batches = np.array(np.split(x,1875,axis=1))
		targets = np.array(np.split(y,1875))
		for i in bar(range(epoch)):
			if i > 650:
			 	rate = 0.0000005
			elif i> 400:
			 	rate = 0.0000025
			else:
			 	rate = 0.000005
				
			r = np.random.permutation(1875)
			batches = batches[r]
			targets = targets[r]
			
			with open("accuracy.csv" , "a") as f:
				t = csv.DictWriter(f , fieldnames = ['cost' , "accuracy",'accuracy_dev'])
			
				for num,(batch,target) in enumerate(zip(batches,targets)):
					
					s = np.random.permutation(batch.shape[1])
					batch = batch[: , s]
					target = target[s , :]

					out = self.forward(x = batch)
					costs.append(self.cost(out,target))
						
					self.backward(y=target,x=batch)
					self.adam(i,rate = rate)
					


				dev_pred = np.argmax((self.forward(devx).T), axis = 1).reshape(1000,1)
				
				pred = np.argmax(out.T, axis = 1).reshape(32,1)
				y = np.argmax(target , axis = 1).T.reshape(32,1)

				t.writerow({'cost' : self.cost(out,target) , 'accuracy' : accuracy(pred,y) , 'accuracy_dev': accuracy(dev_pred,devy)})


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

		db4 =  np.sum(-theta4 , axis = 1 , keepdims = True)/32
		db3 =  np.sum(theta3 , axis = 1 , keepdims = True)/32
		db2 =  np.sum(theta2 , axis = 1 , keepdims = True)/32
		db1 =  np.sum(theta1 , axis = 1 , keepdims = True)/32


		dw4 = np.dot(theta4,a3.T)/32 + ((l/32)*w4).T
		dw3 = np.dot(theta3,a2.T)/32 + ((l/32)*w3).T
		dw2 = np.dot(theta2,a1.T)/32 + ((l/32)*w2).T
		dw1 = np.dot(theta1,x.T)/32 + ((l/32)*w1).T

		self.put(db4=db4,db3=db3,db2=db2,db1=db1)
		self.put(dw4=dw4,dw2=dw2,dw3=dw3,dw1=dw1)

	def update(self,rate = 0.005):
		w1,w2,w3,w4,dw1,dw2,dw3,dw4 = self.get('w1','w2','w3','w4','dw1','dw2','dw3','dw4')
		b1,b2,b3,b4,db1,db2,db3,db4 = self.get('b1','b2','b3','b4','db1','db2','db3','db4')

		w1 -= rate*dw1.T
		w2 -= rate*dw2.T
		w3 -= rate*dw3.T
		w4 -= rate*dw4.T
		

		b1 -= rate*db1
		b2 -= rate*db2
		b3 -= rate*db3
		b4 -= rate*db4

		self.put(b1=b1,b2=b2,b3=b3,b4=b4)
		self.put(w1=w1,w2=w2,w3=w3,w4=w4)


	def adam(self,t , beta1 = 0.9 , beta2 = 0.999 , e = 1e-8 , rate=0.0005):
		vdw = self.get('vdw')
		sdw = self.get('sdw')
		vdb = self.get('vdb')
		sdb = self.get('sdb')

		w1,w2,w3,w4,dw1,dw2,dw3,dw4 = self.get('w1','w2','w3','w4','dw1','dw2','dw3','dw4')
		b1,b2,b3,b4,db1,db2,db3,db4 = self.get('b1','b2','b3','b4','db1','db2','db3','db4')
		dw = [dw1,dw2,dw3,dw4]
		db = [db1,db2,db3,db4]

		#print(np.array(vdw).shape, end = '\n')
		#print(list(map(lambda v: print(len(v)) , vdw)))


		vdw = [(beta1*x + (1-beta1)*y.T) for x,y in zip(*vdw,dw)]
		sdw = [(beta2*x + (1-beta2)*np.square(y.T)) for x,y in zip(*sdw,dw)] 
		vdb = [(beta1*x + (1-beta1)*y) for x,y in zip(*vdb,db)]
		sdb = [(beta2*x + (1-beta2)*np.square(y)) for x,y in zip(*sdb,db)] 

		vdw1,vdw2,vdw3,vdw4 = vdw 
		sdw1,sdw2,sdw3,sdw4 = sdw
		vdb1,vdb2,vdb3,vdb4 = vdb
		sdb1,sdb2,sdb3,sdb4 = sdb
		

		w1 -= rate*((vdw1)/(np.sqrt(sdw1)+e))
		w2 -= rate*((vdw2)/(np.sqrt(sdw2)+e))
		w3 -= rate*((vdw3)/(np.sqrt(sdw3)+e))
		w4 -= rate*((vdw4)/(np.sqrt(sdw4)+e))

		b1 -= rate*((vdb1)/(np.sqrt(sdb1)+e))
		b2 -= rate*((vdb2)/(np.sqrt(sdb2)+e))
		b3 -= rate*((vdb3)/(np.sqrt(sdb3)+e))
		b4 -= rate*((vdb4)/(np.sqrt(sdb4)+e))


		self.put(vdw=vdw,sdw=sdw,vdb=vdb,sdb=sdb)
		self.put(b1=b1,b2=b2,b3=b3,b4=b4)
		self.put(w1=w1,w2=w2,w3=w3,w4=w4)


	def put(self,**kwargs):
		for key,value in kwargs.items():
			self.cache[key] = value

	def get(self,*args):
		u = tuple(map(lambda t: self.cache[t] , args))
		return u


def main():
	net = network(x_train,y_train)

	costs = net.train(epoch = 1000)

	costs = [cost for i,cost in enumerate(costs) if i%500 == 0]
	
	pred = np.argmax((net.forward(x_test).T), axis = 1).reshape(10000,1)
	y = np.argmax(y_test , axis = 1).T.reshape(10000,1)

	pred1 = np.argmax((net.forward().T), axis = 1).reshape(60000,1)
	y1 = np.argmax(y_train , axis = 1).T.reshape(60000,1)

	print(accuracy(pred,y))
	print(accuracy(pred1,y1))
	plt.plot(costs)
	plt.show()

	# plt.show(block=False)
	# #plt.pause(6)
	# #plt.close()



if __name__ == '__main__':
	main()
