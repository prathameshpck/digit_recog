import matplotlib.pyplot as plt
import matplotlib 
from matplotlib.animation import FuncAnimation
import pandas as pd 



fig, (ax1,ax2,ax3) = plt.subplots(nrows = 3 , ncols = 1)

def get_data():
	accu = pd.read_csv('accuracy.csv')
	cost = accu['cost']
	accuracy = accu['accuracy']
	accuracy_dev = accu['accuracy_dev']
	return cost,accuracy,accuracy_dev

while True:
	plt.ion()
	x,y,z = get_data()
	
	line1 ,  = ax1.plot(x)
	line2 ,  = ax2.plot(y)
	line3 ,  = ax3.plot(z)
	#ax2.text(top,top,'Hi' )
	fig.canvas.draw()
	
	plt.pause(0.05)


plt.show(block = False)
exit()