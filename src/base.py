def enum(*sequential, **named):
    # Enum definition for backcompatibility
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Enum to encode the which phase a layer is in at the moment.
Phase = enum('train', 'test', 'validation')

class BaseLayer:
    
    def __init__(self):
        self.phase = Phase.train
        
    def forward(self, x):
        """ Return the result of the forward pass of this layer. Save intermediate results
        necessary to compute the gradients in the backward pass. 
        """
        raise NotImplementedError('Base class - method is not implemented')
    
    def backward(self, error):
        """ Update the parameters/weights of this layer (if applicable), 
        and return the gradient with respect to the input.
        """
        raise NotImplementedError('Base class - method is not implemented')