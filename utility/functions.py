import numpy as np 

def sigmoid(z):
	return 1/(1+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

def relu(z):
	return np.maximum(z,0.001)

def relu_prime(x):
	y = x

	y[y<=0] = 0.001
	y[y>0] = 1

	return y
 	
def hotkey(z):
	hot = []
	for item in z:
		arr = np.zeros(10 , dtype = int)
		arr[item] = 1
		hot.append(arr)

	return np.array(hot)

def softmax(z):
	maxs = np.max(z,axis= 0)

	shift = z - maxs
	#print(shift)
	exp = np.exp(shift)
	sums = np.sum(exp , axis = 0)
	return exp/sums

def accuracy(x,y):
	count = 0 
	for a,b in zip(x,y):
		if a==b:
			count+=1

	return count*100/x.shape[0]




