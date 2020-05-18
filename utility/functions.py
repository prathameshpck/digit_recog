import numpy as np 

def sigmoid(z):
	return 1/(1+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

def relu(z):
	return z*(z>0)

def relu_prime(z):
	return 1*(z>0)


