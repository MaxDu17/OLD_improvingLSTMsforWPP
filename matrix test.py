import numpy as np

x = [1]
y = [2,2,2,2,2,2,2]

#print(np.matmul(x,np.transpose(y)))
print(np.concatenate([x,y], axis = 0))