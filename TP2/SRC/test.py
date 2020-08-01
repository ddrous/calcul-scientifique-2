import numpy as np
import numpy.linalg as npalg
import scipy as sp
import scipy.linalg as lin
import scipy.sparse as spsp
import scipy.sparse.linalg as spsplin


# x = np.zeros((3, 3))
# print(x)

# y = np.ones((3, 1))

# # z = np.stack((x, y), axis=1)
# # print(z)

# x = np.concatenate((x, y), axis=1)
# print(x)

# z = np.concatenate((x, y), axis=1)
# print(z)

# total = 100
 
# def func3():
#     listOfGlobals = globals()
#     listOfGlobals['total'] = 15
#     total = 22
#     print('Local Total = ', total)
 
# print('Total = ', total)
# func3()
# print('Total = ', total)

d = 2
n = d**2
B = spsp.diags([[4.]*n,[-1]*(n-1),[-1] *(n-1),[-1] *(n-d),[-1] *(n-d)],[0,1,-1,d,-d])
B = B.tocsc()

b = np.zeros((n, n))
for i in range(n):
    b[i, i] = 1

M = spsplin.spilu(B)

print("b", b)
print(B.todense())
x = M.solve(b)

print(x)
sol = lin.lu(B.todense())
print("sol")
print(sol[1],"\n", sol[2])
