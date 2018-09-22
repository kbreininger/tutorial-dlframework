class Initializer:
    """ Base class for initializers. """
    def initialize(self, weight_shape):
        """ Return weights initialized according to the subclass definition. 
            Required to work for arbitrary weight shapes.
            Base class. 
        """
        
        # Raises an exeption in base class.
        raise NotImplementedError('Method is not implemented')

        
class Const(Initializer):
    
    def __init__(self, value):
        """ Create a constant initializer.
            params: value (float): constant that is used for initialization of weights
        """
        # TODO: Implement
        pass

    def initialize(self, weight_shape):
        """ Return a new array of weights initialized with a constant value provided by self.value.
            param: weight_shape: shape of the new array
            returns (np.ndarray): array of the given shape
        """
        # TODO: Implement
        pass

class UniformRandom(Initializer):
    
    def initialize(self, weight_shape):
        """ Return a new array of weights initialized by drawing from a uniform distribution with range [0, 1].
            param: weight_shape: shape of new array
            returns (np.ndarray): array of the given shape
        """
        # TODO: Implement
        pass

        
class He(Initializer):
       
    def initialize(self, weight_shape):
        """ Return a new array of weights initialized according to He et al.: Delving Deep into Rectifiers.
            param: weight_shape: shape of the np.array to be returned, the second dimension is assumed to be the 
                   number of input nodes
            returns (np.ndarray): array of the given shape
        """        
        # TODO: Implement
        pass
        