"""This file contains new code for hw1 bonus that you should copy+paste to the appropriate file."""

# ---------------------------------
# nn/functional.py
# ---------------------------------
class Dropout(Function):
    @staticmethod
    def forward(ctx, x, p=0.5, is_train=False):
        """Forward pass for dropout layer.

        Args:
            ctx (ContextManager): For saving variables between forward and backward passes.
            x (Tensor): Data tensor to perform dropout on
            p (float): The probability of dropping a neuron output.
                       (i.e. 0.2 -> 20% chance of dropping)
            is_train (bool, optional): If true, then the Dropout module that called this
                                       is in training mode (`<dropout_layer>.is_train == True`).
                                       
                                       Remember that Dropout operates differently during train
                                       and eval mode. During train it drops certain neuron outputs.
                                       During eval, it should NOT drop any outputs and return the input
                                       as is. This will also affect backprop correspondingly.
        """
        if not type(x).__name__ == 'Tensor':
            raise Exception("Only dropout for tensors is supported")
        
        raise NotImplementedError("TODO: Implement Dropout(Function).forward() for hw1 bonus!")
        
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("TODO: Implement Dropout(Function).backward() for hw1 bonus!")
