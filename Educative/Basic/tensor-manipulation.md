#### PyTorch Tensor Manipulation for Machine Learning

Course URL: https://www.educative.io/courses/pytorch-tensor-manipulation/about-this-course?openHLOPage=true



###### Notable PyTorch Consumers

Tesla Autopilot
Uber's Pyro
HuggingFace Transformers





###### What is a PyTorch tensor

In short, a PyTorch tensor is an n-dimensional array that is the same as a NumPy array or TensorFlow tensor. You can consider a rank 0 tensor as a scalar, a rank 1 tensor as a
vector, and a rank 2 tensor as a matrix. For higher-dimensional, they are rank n tensor. 

Tensors form the fundamental building block of the PyTorch ecosystem. Various components, such as network layers, loss functions, gradient descent, and optimizers rely on underlying tensor operations.

Rank 0 Tensor - scalar
Rank 1 Tensor - vector
Rank 2 Tensor - matrix
Rank 3 Tensor - matrices
Rnak 4 Tensor - vector of matrices


###### Creating a Tensor from a List
Creating a tensor from a list or a nested list is easy.
Import the torch library and call the tensor function.

```python

import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([1], [2], [3])

print(a)

print(b)


```


###### Creating a tensor from a NumPy array

```python

import torch
import numpy as np

na = np.array([1, 2, 3])
a = torch.tensor(na)
b = torch.from_numpy(na)
print(a)
print(b)

```






###### Creating Special Tensors
* Identity tensors
* Ones tensors
* Zeros tensors


PyTorch provides some useful functions to create special tensors, such as the identity tensors having all zeros or ones.

*   eye() creates an identity tensor with an integer.
*   zeros() creates a tensor with all zeros, the parameter
    could be an integer or a tuple that defines the shape
    of the tensor.
*   ones() creates a tensor with all ones like ones.
    The parameter could be an integer or a tuple that defines
    the shape of the tensor.


```python
import torch

# Create an identity tensor with 3*3 shape.
identity = torch.eye(3)
print(identity)

# Create a tensor with 2*2 shape whose values are all 1.
ones = torch.ones((2, 2))
print(ones)

# Create a tensor with 3*3 shape whose values are all 0.
zeros = torch.zeros((3, 3))

```


###### Creating a Random Tensor
PyTorch provides some useful functions to create a tensor with a random value.

* rand() - It creates a tensor filled with random numbers from a uniform distribution. The parameter is a sequence of
integers defining the shape of the output tensor. It can be
a variable number of arguments or a collection like a list or
a tuple.
* randn() - It creates a tensor filled with random numbers
  from a normal distribution with mean 0 and variance 1. The
  parameter is the same as the rand().
* randint() - Unlike the functions above, this function
  creates a tensor with integer values with low, high and size parameters. 
  * low is an optional with a default 0 value, which means the lowest value.
  * high means the highest value.
  * size is a tuple that defines the shape of the tensor.

```python
import torch

# Create a tensor with a 1*10 shape with a random value
# between 0 and 1.
r0 = torch.rand(10)

print(r0)
print("************************************************")


# Create a tensor with 10*1 shape with random value 
# between 0 and 1.
r1 = torch.rand((10, 1))
print(r1)
print("************************************************")


# Create a tensor with 2*2 shape with random value
# between 0 and 1.
r2 = torch.rand((2, 2))
print(r2)
print("************************************************")


# Create a tensor with 2*2 shape with random value
# from a normal distribution.
r3 = torch.randn((2, 2))
print(r3)
print("************************************************")


# Create an integer type tensor with 3*3 shape with random
# value between 0 and 10.
r4 = torch.randint(high=10, size=(3, 3))
print(r4)
print("************************************************")


# Create an integer type tensor with 3*3 shape with random
# value between 5 and 10.
r5 = torch.randint(low=5, high=10, size=(3, 3))
print(r5)

```


###### Creating a range tensor.
PyTorch also provides a function arange that generates values
in (start; end), like NumPy.


```python
import torch

a = torch.arange(1, 10)
print(a)

```


#### Tensor Metadata

###### Getting type from dtype
The dtype attribute of a PyTorch tensor can be used to
get its type information.

The code below creates a tensor with the float type and
prints the type information from dtype. You can try the
code at the end of this lesson.


```python
import torch
a = torch.tensor([1, 2, 3], dtype=torch.float)
print(a.dtype)
```


###### Getting the number of dim
As shown in the code below, the number of dimensions of a
tensor in PyTorch can be obtained using the attribute ndim
or using the function dim() or its alias ndimensions().

```python
import torch

a = torch.ones((3, 4, 6))
print(a.ndim)
print(a.dim())

```


###### Getting the number of elements
PyTorch provides two ways to get the number of elements of
a tensor, nelement() and numel(). Both of them are functions.

```python
import torch

a = torch.ones((3, 4, 6))
print(a.numel())

```