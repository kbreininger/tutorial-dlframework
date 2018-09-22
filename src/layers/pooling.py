class MaxPoolLayer(BaseLayer):
    
    def __init__(self, neighborhood=(2, 2), stride=(2, 2)):
        """ Max pooling layer.
           param: neighborhood: tuple with shape (sp, sq) which denote the kernel size of the pooling operation in 
           the spatial dimensions
           param: stride: tuple with shape (np, nq) which denote the subsampling factor of the pooling operation in
           the spacial dimensions
        """
        # TODO: define necessary class variables
        pass
    
    def forward(self, x):
        """ Return the result of maxpooling on the input.
            param: x (np.ndarray) with shape (b, n_channels, p, q) where b is the batch size, 
                   n_channels is the number of input channels and p x q is the image size
            returns (np.ndarray): the result of max pooling, of shape (b, n_channels, p', q')
                   where b is the batch size, n_channels is the number of input channels and 
                   p' x q' is the new image size reduced by the stride. 
        """
        # TODO: Implement forward pass of max pooling
        pass
    
    def backward(self, error):
        """ Return the gradient with respect to the previous layer.
            param: error(np.ndarray): the gradient passed own from the subsequent layer, 
                   of shape [b, n_channels, p', q'] where b is the batch size, n_channels is the 
                   number of channels and p' x q' is the image size reduced by the stride
            returns (np.ndarray): the gradient w.r.t. the previous layer, of shape [b, n_channels, p, q] 
                   where b is the batch size, n_channels is the number of input channels to this layer and 
                   p x q is the image size prior to downsampling.
        """
        # TODO: Implement backward pass of max pooling
        pass