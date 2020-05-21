import numpy as np 

def sigmoid(z):
	return 1/(1+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

def relu(z):
	return np.maximum(z,0)

def relu_prime(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
 	
def hotkey(z):
	hot = []
	for item in z:
		arr = np.zeros(10 , dtype = int)
		arr[item] = 1
		hot.append(arr)

	return np.array(hot)

def softmax(z):
	shift = z - np.max(z)
	#print(shift)
	exp = np.exp(shift)

	return exp/np.sum(exp, axis = 0)

