import matplotlib.pyplot as plt
import matplotlib 
from matplotlib.animation import FuncAnimation
import pandas as pd 



fig, (ax1,ax2) = plt.subplots(nrows = 2 , ncols = 1)

def get_data():
	accu = pd.read_csv('accuracy.csv')
	cost = accu['cost']
	accuracy = accu['accuracy']
	return cost,accuracy

while True:
	plt.ion()
	x,y = get_data()
	
	top = y.to_numpy()[-1]
	line1 ,  = ax1.plot(x)
	line2 ,  = ax2.plot(y)
	#ax2.text(top,top,'Hi' )
	fig.canvas.draw()
	
	plt.pause(0.05)


plt.show(block = False)
exit()