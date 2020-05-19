import os,codecs 
import numpy as np 

datapath = './dataset/'

files = os.listdir(datapath)

def get_int(b):
	return int(codecs.encode(b , 'hex'), 16)

def load():

	datapath = './dataset/'
	files = os.listdir(datapath)
	datas = {}

	for file in files:
		if(file.endswith('ubyte')):
			print("Processing " , file)
			with open(datapath+file , 'rb') as f:
				data = f.read()

				magic = get_int(data[:4])
				num = get_int(data[4:8])

				if(magic == 2051):
					category = 'images'
					row = get_int(data[8:12])
					cols = get_int(data[12:16])

					parsed = np.frombuffer(data,dtype = np.uint8 , offset = 16)
					buffers = parsed.reshape(num,row,cols)

				elif(magic == 2049):
					category = 'labels'

					parsed = np.frombuffer(data,dtype = np.uint8 , offset = 8)
					buffers= parsed.reshape(num)
				
				
				if(num == 10000):
					set = 'test'
				elif(num ==60000):
					set = 'train'

				datas[set+' '+category] = buffers

	return datas['train images'] , datas['train labels'] , datas['test images'] , datas['test labels']