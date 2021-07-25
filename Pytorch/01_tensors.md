# Tensors

tensor 是 PyTorch 中特別的 data structure, 類似於 Numpy 中的 ndarray, 差別是 tensor 可以跑在 GPU or hardware accelerators 上, 而 tensor 可以與 Numpy array 共享同塊記憶體而不用再 copy data, 且 tensor 在 automatic differentiation 還有經過優化

## init

### 1. Directly from data

```python
import numpy as np
import torch

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

print(x_data)
print(type(x_data))
```

output: 

```
tensor([[1, 2],
        [3, 4]])
<class 'torch.Tensor'>
```

### 2. From a NumPy array

```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

### 3. From another tensor

```python
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
```

### 4. With random or constant values

```python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

## attribute

每個 tensor 皆有下列屬性:

1. shape
    * the number of elements in each dimension
    * ```.shape``` vs. ```.size()```:
        * ans: ```.shape``` is an alias for ```.size()```, and was added to more closely match numpy [the issue from github](https://github.com/pytorch/pytorch/issues/5544)
2. dtype
    * data type of a ```torch.Tensor```, Pytorch support 12 dufferebt data types now
3. device
    * an object representing the device on which a ```torch.Tensor``` is or will be allocated.
    * **By default, tensors are created on the CPU**, We need to explicitly move tensors to the GPU using ```.to``` method:
        
        ```python
        # We move our tensor to the GPU if available
        if torch.cuda.is_available():
        tensor = tensor.to('cuda')
        ```

```python
import numpy as np
import torch

tensor = torch.rand(3, 4)

print(tensor)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

output:

```
tensor([[0.7658, 0.1360, 0.3979, 0.8463],
        [0.9995, 0.5778, 0.2965, 0.5668],
        [0.2671, 0.1712, 0.9587, 0.8910]])
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

more in [Docs: TENSOR ATTRIBUTES](https://pytorch.org/docs/stable/tensor_attributes.html)

## operations