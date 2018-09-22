class FullyConnectedLayer(BaseLayer):
    def __init__(self, input_size, output_size, learning_rate):
        """ A fully connected layer.
            param: input_size (int): dimension n of the input vector
            param: output_size (int): dimension m of the output vector
            param: learning_rate (float): the learning rate of this layer
        """
        # TODO: define the neccesary class variables
        pass

    def forward(self, x):
        """ Compute the foward pass through the layer.
            param: x (np.ndarray): input with shape [b, n] where b is the batch size and n is the input size
            returns (np.ndarray): result of the forward pass, of shape [b, m] where b is the batch size and
                   m is the output size
        """
        # TODO: Implement forward pass of the fully connected layer
        # Hint: Think about what you need to store during the forward pass to be able to compute 
        # the gradients in the backward pass 
        pass
    
    def get_gradient_weights(self):
        """ 
        returns (np.ndarray): the gradient with respect to the weights and biases from the last call of backward(...)
        """
        # TODO: Implement 
        pass
    
    def backward(self, error):
        """ Update the weights of this layer and return the gradient with respect to the previous layer.
            param: error (np.ndarray): of shape [b, m] where b is the batch size and m is the output size
            returns (np.ndarray): the gradient w.r.t. the previous layer, of shape [b, n] where b is the 
                   batch size and n is the input size
        """
        # TODO: Implement backward pass of the fully connected layer
        # Hint: Be careful about the order of applying the update to the weights and the calculation of 
        # the error with respect to the previous layer.
        pass
    
    def initialize(self, weights_initializer, bias_initializer):
        """ Initializes the weights/bias of this layer with the given initializers.
            param: weights_initializer: object providing a method weights_initializer.initialize(weights_shape)
                   which will return initialized weights with the given shape
            param: bias_initializer: object providing a method bias_initializer.initialize(bias_shape) 
                   which will return an initialized bias with the given shape
        """
        # TODO: Implement
        pass