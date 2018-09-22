class DropOut(BaseLayer):
    
    def __init__(self, probability):
        """ DropOut Layer.
            param: probability: probability of each individual activation to be set to zero, in range [0, 1]    
        """
        # TODO: Implement initialization
        
        pass
    
    def forward(self, x):
        """ Forward pass through the layer: Set activations of the input randomly to zero.
            param: x (np.ndarray): input
            returns (np.ndarray): a new array of the same shape as x, after dropping random elements
        """
        # TODO: Implement forward pass of the Dropout layer
        # Hint: Make sure to treat training and test phase accordingly.
        pass
    
    def backward(self, error):
        """ Backward pass through the layer: Return the gradient with respect to the input.
            param: error (np.ndarray): error passed down from the subsequent layer, of the same shape as the 
                   output of the forward pass
            returns (np.ndarray):  gradient with respect to the input, of the same shape as error
        """
        # TODO: Implement backward pass of the Dropout layer
        pass