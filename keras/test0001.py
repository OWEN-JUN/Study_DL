import numpy as np

a=[[1,2],[2,7],[3,2]]
b=[[2,4],[3,6],[4,3]]
b1 = [[[5],[6],[7],[8]]]
c = np.array([a,b])
a=[[1,2],[2,7],[3,2]]
b=[[2,4],[3,6],[4,3]]
b1 = np.array(b1).reshape(1,-1)
print(b1)
b1 = np.array(b1).reshape(-1,1)
print(b1)
c = c.reshape(-1,1)
print(c)


