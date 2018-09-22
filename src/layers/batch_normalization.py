class BatchNorm(BaseLayer):
    def __init__(self, learning_rate, convolutional=False):
        """ Batch normalization layer.
            param: learning_rate (float): the learning rate of this layer
            param: convolutional(boolean): if true, only a scalar mean and a scalar variance is 
                   calculated for every channel, otherwise mean and variance have the same dimension 
                   as the input
        """
        # TODO: Implement initialization
        pass

    def forward(self, x):
        """ Return the batch normalized input.
            param: x(np.ndarray): input, of arbitrary shape
            returns (np.ndarray): result of batch normalization, of the same shape as x
        """
        # TODO: Implement forward pass of the batch normalization layer
        
        # Hint 1: Make sure to treat training and test phase accordingly.
        # Hint 2: If the network has never seen any training data, but is applied in "test mode", the network 
        #         should not change the distribution of the input. Initialize the respective variable after the
        #         first training input is received.
        pass

    def backward(self, error):
        """ Return the gradient with respect to the previous layer.
            param: error(np.ndarray): error passed down from the subsequent layer, of the same shape as the input
                   in the forward pass
            returns (np.ndarray): gradient w.r.t. the input, of the same shape as error
        """
        # TODO: Implement backward pass of the batch normalization layer
        pass

    def get_gradient_weights(self):
        """ Returns the gradient with respect to the weights, i.e. \gamma, from the last call of backward() """
        # TODO: Implement
        pass

    def get_gradient_bias(self):
        """ Returns the gradient with respect to the bias, i.e. \beta, from the last call of backward() """ 
        # TODO: Implement
        pass
