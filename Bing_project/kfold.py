import numpy as np

def blah(arr,k):
	out = []	
	#arr=np.zeros(shape=(5,2))
	#k=3
	x = np.array_split(arr,k)
	for i in range(0,k):
		temp = x[:i]+x[(i+1):]
		np.array(temp[0])
		np.array(temp[1])
		tup = (x[i],np.concatenate((temp[0],temp[1])))
		np.array(tup)
		out.append(tup)
	np.array(out)
	#print out
	return out
