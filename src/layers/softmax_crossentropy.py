class SoftMaxCrossEntropyLoss(BaseLayer):
    
    def forward(self, x, labels):
        """ Return the cross entropy loss of the input and the labels after applying the softmax to the input. 
            param: x (np.ndarray): input, of shape [b, k] where b is the batch size and k is the input size
            param: labels (np.ndarray): the corresponding labels of the training set in one-hot encoding for 
                   the current input, of the same shape as x
            returns (float): the loss of the current prediction and the label
        """
        # Todo: Implement forward pass
        pass
    
    def backward(self, labels):
        """ Return the gradient of the SoftMaxCrossEntropy loss with respect to the previous layer.
            param: labels (np.ndarray): (again) the corresponding labels of the training set for the current input, 
                   of shape [b, k] where b is the batch size and k is the input size
            returns (np.ndarray): the error w.r.t. the previous layer, of shape [b, k] where b is the batch 
                   size and n is the input size
        """
        # TODO: Implement backward pass
        pass
    
    def predict(self, x):
        """ Return the softmax of the input.  This can be interpreted as probabilistic prediction of the class.
            param: x (np.ndarray): input with shape [b, k], where b is the batch size and n is the input size
            returns (np.ndarray): the result softmax(x), of the same shape as x
        """
        # TODO: Implement softmax
        pass