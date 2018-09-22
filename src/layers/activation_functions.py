class Sigmoid(BaseLayer):
    
    def forward(self, x):
        """ Return the element-wise sigmoid of the input.
            param: x (np.ndarray): input to the activation function, of arbitrary shape
            returns (np.ndarray): element-wise sigmoid(x), of the same shape as x
        """
        # TODO: Implement forward pass of the Sigmoid
        pass
        
    def backward(self, error):
        """ Return the gradient with respect to the input.
            param: error (np.ndarray): the gradient passed down from the subsequent layer, of the same 
                   shape as x in the forward pass
            returns (np.ndarray): the gradient with respect to the previous layer, of the same shape as error 
        """
        # TODO: Implement backward pass of the Sigmoid
        pass
    

class ReLU(BaseLayer):
    
    def forward(self, x):
        """ Return the result of a ReLU activation of the input.
            param: x (np.ndarray): input to the activation function, of arbitrary shape
            returns (np.ndarray): element-wise ReLU(x), of the same shape as x
        """
        # TODO: Implement forward pass of the ReLU
        pass
    
    def backward(self, error):
        """ Return the gradient with respect to the input.
            param: error (np.ndarray): the gradient passed down from the previous layer, arbitrary shape (same as x)
            returns (np.ndarray): gradient with respect to the input, of the same shape as error 
        """
        # TODO: Implement backward pass of the ReLU
        pass