import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import pandas as pd 


fig, (ax1,ax2) = plt.subplots(nrows = 2 , ncols = 1)

def animate(i):
	accu = pd.read_csv('accuracy.csv')
	cost = accu['cost']
	accuracy = accu['accuracy']
	print(i)
	plt.cla()
	line1 ,  = ax1.plot(cost)
	line2 ,  = ax2.plot(accuracy)
	plt.tight_layout()
	
	return (line1 , line2)

ani = FuncAnimation(plt.gcf() , animate , blit = True , interval = 200)
plt.show()