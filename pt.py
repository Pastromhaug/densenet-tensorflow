import numpy as np
a = range(1)
num = (int(np.log(len(a))/np.log(2)))
b = [-(2**i) for i in range(num+1)]
c = [a[i] for i in b]
print(a)
print(num)
print(b)
print(c)
