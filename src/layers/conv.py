class FlattenLayer(BaseLayer):
    def __init__(self):
        # TODO: define the necessary class variables
        pass
    
    def forward(self, x):
        """ Return a flattened version of the input.
            param: x (np.ndarray): input, of shape [b, n_channels, p, q] where b is the batch size, 
                   n_channels is the number of channels and p x q is the image size
            returns (np.ndarray): a flattened representation of x of shape [b, v] 
                   where b is the batch size and v is the output size = n_channels * p * q
        """
        # TODO: Implement flattening of the image
        pass
    
    def backward(self, error):
        """ Return the gradient with respect to the input.
            param: error (np.ndarray): the gradient passed down from the subsequent layer, of shape [b, m],
                   where b is the batch size and m is the output size with m = n_channels * p * q from 
                   the forward pass
            returns (np.ndarray): the error with restored dimensions from the forward pass, i.e. with 
                   shape [b, n_channels, p, q] where b is the batch size, n_channels is the number of 
                   channels and p x q is the image size
        """
        # TODO: Restore the image dimensions
        pass


class ConvolutionalLayer(BaseLayer):
    
    def __init__(self, stride, kernel_shape, n_kernels, learning_rate):
        """ 
            param: stride: tuple in the form of (np, nq) which denote the subsampling factor of the 
                   convolution operation in the spatial dimensions
            param: kernel_shape: integer tuple in the form of (n_channels, m, n) where n_channels is 
                   the number of input channels and m x n is the size of the filter kernels
            param: n_kernels (int): number of kernels and therefore the number of output channels
            param: learning_rate (float): learning rate of this layer
        """
        # TODO: define the neccesary class variables
        pass 
    
    def forward(self, x):
        """ Return the result of the forward pass of the convolutional layer.
            param: x(np.ndarray): input, of shape [b, n_channels, p, q],  where b is the batch size, 
                   n_channels is the number of input channels and p x q is the image size
            returns (np.ndarray): result of the forward pass, of shape (b, n_kernels, p', q') 
                   where b is the batch size, n_kernels is the number of kernels in this layer and 
                   p' x q' is the output image size (which depends on the stride)
        """
        # TODO: Implement forward pass of the convolutional layer
        pass
    
    def backward(self, error):
        """ Update the weights of this layer and return the gradient with respect to the input.
            param: error (np.ndarray): of shape (b, n_kernels, p', q') where b is the batch size, n_kernels
                   is the number of kernels and p' x q' is the spacial error size (depends on the stride)
            returns (np.ndarray): the gradient with respect to the input, of shape (b, n_channels, p, q) 
                   where b is the batch size, n_channels is the number of input channels to this layer and 
                   p x q is the image size.
        """ 
        # TODO: Implement backward pass of the convolutional layer
        pass
    
    def get_gradient_weights(self):
        """ Returns the gradient with respect to the weights from the last call of backward() """
        # TODO: Implement
        pass

    def get_gradient_bias(self):
        """ Returns the gradient with respect to the bias from the last call of backward() """
        # TODO: Implement
        pass
    
    def initialize(self, weights_initializer, bias_initializer):
        """ Initializes the weights/bias of this layer with the given initializers.
            param: weights_initializer: object providing a method weights_initializer.initialize(weights_shape)
                   which will return initialized weights with the given shape
            param: bias_initializer: object providing a method bias_initializer.initialize(bias_shape) 
                   which will return an initialized bias with the given shape
        """
        # TODO: Implement. To make sure that He initialization works as intended, make sure the second dimension 
        # of weights_shape contains the number of input nodes that can be computed as n_in = n_channels * m * n
        # and reshape the weights to the correct shape afterwards.
        pass