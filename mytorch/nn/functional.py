import numpy as np

import mytorch.tensor as tensor
from mytorch.autograd_engine import Function

def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                                    is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.T)

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                                                 is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data / a.data)

"""EXAMPLE: This represents an Op:Add node to the comp graph.

See `Tensor.__add__()` and `autograd_engine.Function.apply()`
to understand how this class is used.

Inherits from:
    Function (autograd_engine.Function)
"""
class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape
        #if a.data.shape != b.data.shape:
            #raise Exception("Both args must have same sizes: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor') or a.data.shape != b.data.shape:
            raise Exception("Both args must be tensors and have same size")

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a,b)
        
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data - b.data, requires_grad=requires_grad,is_leaf=not requires_grad)
        return c 

    #raise Exception("TODO: Implement '-' forward")


    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = np.ones(a.shape) * grad_output.data
        grad_b = - np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        #grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        #grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

#raise Exception("TODO: Implement '-' backward")

class Mul(Function):
    @staticmethod
    def forward(ctx,a,b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape
        #if a.data.shape != b.data.shape:
            #raise Exception("Both args must have same sizes: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a,b)

        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data * b.data, requires_grad=requires_grad,is_leaf=not requires_grad)
        return c 

    @staticmethod
    def backward(ctx,grad_output):
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = b.data * grad_output.data
        grad_b = a.data * grad_output.data

        # the order of gradients returned should match the order of the arguments
        #grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        #grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


class Div(Function):
    @staticmethod
    def forward(ctx,a,b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape
        #if a.data.shape != b.data.shape:
            #raise Exception("Both args must have same sizes: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a,b)

        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data / b.data, requires_grad=requires_grad,is_leaf=not requires_grad)
        return c 

    @staticmethod
    def backward(ctx,grad_output):
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = (np.ones(b.shape) / b.data) * grad_output.data
        grad_b = - (a.data / (b.data**2)) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        #grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        #grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only log of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        if axis is not None:
            ctx.len = a.shape[axis]
        ctx.keepdims = keepdims
        requires_grad = a.requires_grad
        print(a)
        c = tensor.Tensor(a.data.sum(axis = axis, keepdims = keepdims), \
                          requires_grad=requires_grad, is_leaf=not requires_grad)
        #print(a.shape, c.shape)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.data

        if (ctx.axis is not None) and (not ctx.keepdims):
            grad_out = np.expand_dims(grad_output.data, axis=ctx.axis)
        else:
            grad_out = grad_output.data.copy()

        grad = np.ones(ctx.shape) * grad_out

        assert grad.shape == ctx.shape
        # Take note that gradient tensors SHOULD NEVER have requires_grad = True.
        return tensor.Tensor(grad), None, None


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        requires_grad = a.requires_grad or b.requires_grad
        ctx.save_for_backward(a,b)
        c = tensor.Tensor(np.matmul(a.data,b.data), requires_grad=requires_grad,is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.data
        a, b = ctx.saved_tensors
        grad_a = np.matmul(grad_output.data,b.data.T)
        grad_b = np.matmul(a.data.T,grad_output.data)
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


class ReLu(Function):
    @staticmethod
    def forward(ctx,a):
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.maximum(0,a.data), requires_grad=requires_grad,is_leaf=not requires_grad)
        return c
    @staticmethod
    def backward(ctx,grad_output):
        a = ctx.saved_tensors[0]
        a.data[a.data>=0] = 1
        b = np.maximum(0,a.data)
        grad_a = b*grad_output.data
        return tensor.Tensor(grad_a)

# TODO: Implement more Functions below

class cross_entropy(Function):
    @staticmethod
    def forward(ctx,predicted,target):
        ctx.save_for_backward(predicted,target)
        LogSoftmax = np.log((np.exp(predicted.data)) / (np.sum(np.exp(predicted.data),axis=1 ).reshape(-1,1) ) )
        one_hot = to_one_hot(target.data,LogSoftmax.shape[-1])
        N = LogSoftmax.shape[0]
        LLLoss = np.sum(LogSoftmax*one_hot.data)/N
        requires_grad = predicted.requires_grad
        return tensor.Tensor(-LLLoss, requires_grad=requires_grad,is_leaf=not requires_grad)
    @staticmethod
    def backward(ctx,grad_output):
        predicted,target = ctx.saved_tensors
        Softmax = ((np.exp(predicted.data)) / (np.sum(np.exp(predicted.data),axis=1 ).reshape(-1,1) )  ) 
        one_hot = to_one_hot(target.data,Softmax.shape[-1])
        grad_predicted = ((Softmax - one_hot.data)*grad_output.data)/predicted.data.shape[0]
        return (tensor.Tensor(grad_predicted),)


#def cross_entropy(predicted, target):#
    """Calculates Cross Entropy Loss (XELoss) between logits and true labels.
    For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

    Args:
        predicted (Tensor): (batch_size, num_classes) logits
        target (Tensor): (batch_size,) true labels

    Returns:
        Tensor: the loss as a float, in a tensor of shape ()
    """
    #batch_size, num_classes = predicted.shape

    # Tip: You can implement XELoss all here, without creating a new subclass of Function.
    #      However, if you'd prefer to implement a Function subclass you're free to.
    #      Just be sure that nn.loss.CrossEntropyLoss calls it properly.

    # Tip 2: Remember to divide the loss by batch_size; this is equivalent
    #        to reduction='mean' in PyTorch's nn.CrossEntropyLoss

    #raise Exception("TODO: Implement XELoss for comp graph")

def to_one_hot(arr, num_classes):
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

    Example:
    >>> to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [1, 0, 0]]
     
    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    #arr = arr.data.astype(int)

    arr = arr.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad = True)



