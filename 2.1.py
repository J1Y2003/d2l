#2.1.1 Getting Started
import torch

x = torch.arange(12, dtype=torch.float32)
X = x.reshape(-1,4)


#print(torch.zeros(2, 3, 4))
#print(torch.ones(2, 3, 4))

#print(torch.randn(3, 4)) #samples drawn from N(0,1)

#print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])) # outer list is axis 0, inner is 1

#2.1.2 Indexing and Slicing
#print(X)
#print(X[-1])
#print(X[1:3]) #note that this slices along axis 0
X[1, 2] = 17 #we can also edit using indices
X[:2, :] = 12 #we can also assign multiple elements the same value
#print(X)

#2.1.3 Operations
#print(torch.exp(x))

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
#print(x + y, x - y, x * y, x / y, x ** y, sep="\n")

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
"""
print(torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=0).shape) #concatenating along axis 0, so size should be: [6,4]
print(torch.cat((X, Y), dim=1), torch.cat((X, Y), dim=1).shape) #concatenating along axis 1, so size should be: [3,8]
"""
#print(X == Y) #this will create a tensor where each element is either 1/0 depending on Xi == Yi.
#print(X.sum()) #note that this is still a tensor!

#2.1.4 Broadcasting
a = torch.arange(6).reshape((3, 1, 2))
b = torch.arange(4).reshape((1, 2, 2))
print(a, b, sep="\n")
print(a + b) #columns of a is repeated, rows of b is repeated to both form a (3,2) matrix

#2.1.5 Saving Memory
before = id(Y) #previous memory location of Y
Y = Y + X
#print(id(Y) == before)
#We use X[:] = ? to make sure we allocate the new value into the previously allocated array of the variable.
Z = torch.zeros_like(Y) #zeros_like initializes a array of the same shape as Y, where all elements = 0
#print('oldId(Z):', id(Z))
Z[:] = X + Y
#print('newId(Z):', id(Z))

before = id(X)
X += Y
#print(before == id(X)) #this returns True, as +=, -=, etc operator preserves the memory location.

#2.1.6 Conversion to Other Python Objects
A = X.numpy() #converting from torch tensor to numpy tensor
B = torch.from_numpy(A) #vice versa
#print(type(A), type(B))

a = torch.tensor([3.5])
#print(a, a.item(), float(a), int(a)) #either item function of built-in functions work to convert size-1 tensor to Python scalar