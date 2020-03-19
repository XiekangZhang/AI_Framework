import torch

if __name__ == '__main__':
    print(torch.__version__, torch.get_default_dtype())

    tensor_arr = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(tensor_arr, torch.numel(tensor_arr), torch.is_tensor(tensor_arr))

    # tensor takes one argument, Tensor takes more arguments
    tensor_uninitialized = torch.Tensor(2, 2)
    print('using torch.Tensor to create tensor: {}'.format(tensor_uninitialized))
    # to create randomly value tensor
    tensor_initialized = torch.rand(2, 2)
    print('using torch.rand to create tensor: {}'.format(tensor_initialized))

    tensor_int = torch.tensor([5, 3]).type(torch.IntTensor)
    print('change tensor type to int: {}'.format(tensor_int))

    tensor_short = torch.ShortTensor([1.0, 2.0, 3.0])
    print('short type: {}'.format(tensor_short))

    tensor_float = torch.tensor([1.0, 2.0, 3.0]).type(torch.half)
    print('half type: {}'.format(tensor_float))
    # Tensor is split to tensor and empty

    tensor_fill = torch.full((2, 6), fill_value=10)
    print('create a tensor with (2, 6) matrix: ')
    print(tensor_fill)

    tensor_of_ones = torch.ones([2, 4], dtype=torch.int32)
    print('A tensor of size (2, 4) containing all ones: ')
    print(tensor_of_ones)

    tensor_of_zeros = torch.zeros_like(tensor_of_ones)
    print('A tensor of size (2, 4) containing all zeros: ')
    print(tensor_of_zeros)

    tensor_eye = torch.eye(5)
    print('Creating an identity 5*5 tensor: ')
    print(tensor_eye)

    non_zero = torch.nonzero(tensor_eye)
    print('return the index for non-zero elements based on tensor_eye: {}'.format(non_zero))

    print('create a sparse tensor using coordinates specified by indices and values')
    i = torch.tensor([
        [0, 1, 1],
        [2, 2, 0]
    ])
    v = torch.tensor([3, 4, 5], dtype=torch.float32)
    sparse_tensor = torch.sparse_coo_tensor(i, v, [2, 5])
    print(sparse_tensor.data)

    print('.fill_ is in-place operation and it does not have any out-place equivalent')
    initial_tensor = torch.rand(2, 3)
    print('initial tensor created by torch.rand with size of (2, 3): {}'.format(initial_tensor))
    print('using in-place operation: {}'.format(initial_tensor.fill_(10)))
    # print('using out-place operation: {}'.format(initial_tensor.fill(10)))

    # the add() method does an out-of-place add operation and returns a new tensor
    new_tensor = initial_tensor.add(5)
    print('the new tensor {} vs. the old tensor {}'.format(new_tensor, initial_tensor))
    # thee add_ method does an in-place add, changing the calling tensor
    initial_tensor.add_(8)
    new_tensor.sqrt_()
    print(initial_tensor, new_tensor)

    # Indexing, Slicing, Joining, Mutating Ops
    x = torch.linspace(start=0.1, end=10.0, steps=15)
    print(x)

    # Splits a tensor into a specific number of chunks
    tensor_chunk = torch.chunk(x, 3, 0)
    print('after chunk: %s' % str(tensor_chunk))  # splits tensor into 3 parts

    # Concatenates the sequence of tensors along the given dimension
    tensor1 = tensor_chunk[0]
    tensor2 = tensor_chunk[1]
    tensor3 = torch.tensor([3.0, 4., 5.])
    print(torch.cat((tensor1, tensor2, tensor3), 0))

    random_tensor = torch.tensor([
        [10, 8, 30],
        [40, 5, 6],
        [12, 2, 21]
    ])
    print(random_tensor, random_tensor[0, 1], random_tensor[1:, 1:])

    # splits the tensor into chunks
    random_tensor_split = torch.split(random_tensor, 2)
    print(random_tensor_split)

    # view does not create a deep copy - just a view
    print(random_tensor.size())
    resized_tensor = random_tensor.view(9)
    print(resized_tensor, resized_tensor.size())
    # resized_tensor = random_tensor.view(-1, 6)
    # print(resized_tensor)
    random_tensor[2, 2] = 100.0
    print(resized_tensor)

    # Unsqueeze returns a new tensor with a dimension of size one inserted at the specified position
    print(random_tensor, random_tensor.shape)
    tensor_unsqueeze = torch.unsqueeze(random_tensor, 2)  # at the second dimension
    print(tensor_unsqueeze, tensor_unsqueeze.shape)

    # Transpose returns a tensor that is a transposed version of input.
    print(initial_tensor)
    tensor_transpose = torch.transpose(initial_tensor, 0, 1)
    print(tensor_transpose)

    # Sorting tensors:
    # Tensors can be sorted along a specified dimension. If no dimension is specified,
    # the last dimension is picked by default.
    print(random_tensor)
    sorted_tensor, sorted_indices = torch.sort(random_tensor)
    print(sorted_indices, sorted_tensor)

    # Math Operations:
    tensor_float = torch.FloatTensor([-1.1, -2.2, 3.3])
    print(tensor_float)
    # Absolute values
    tensor_abs = torch.abs(tensor_float)
    print(tensor_abs)
    print(initial_tensor)
    new_tensor = torch.add(initial_tensor, 2)
    print(new_tensor)

    rand1 = torch.abs(torch.randn(2, 3))
    rand2 = torch.abs(torch.randn(2, 3))
    add1 = rand1 + rand2
    print(add1, torch.add(rand1, rand2))

    tensor = torch.tensor([
        [-1, -2, -3],
        [1, 2, 3]
    ], dtype=torch.float32)
    tensor_div = torch.div(tensor, tensor + 0.3)
    print(tensor_div)
    tensor_mul = torch.mul(tensor, tensor)
    print(tensor_mul)

    tensor_clamp = torch.clamp(tensor, min=-0.2, max=2)
    print(tensor_clamp)

    # Vector Multiplication
    # Dot production
    t1 = torch.tensor([1, 2])
    t2 = torch.tensor([10, 20])
    dot_product = torch.dot(t1, t2)
    print(dot_product)

    matrix = torch.tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])  # 2*3
    vector = torch.tensor([0, 1, 2])  # 3*1
    matrix_vector = torch.mv(matrix, vector)
    print(f'{matrix.shape} * {vector.shape} = {matrix_vector}')

    another_matrix = torch.tensor([
        [10, 30],
        [20, 0],
        [0, 50]
    ])  # 3*2
    matrix_mul = torch.mm(matrix, another_matrix)
    print(matrix_mul)

    # Returns the indices of the max values of a tensor across a dimension
    print(torch.argmax(matrix_mul, dim=1))
    print(torch.argmin(matrix_mul, dim=0))
