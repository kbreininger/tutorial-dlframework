import numpy
from src.base import Phase
# Nothing to do in this cell: Just make yourself familiar with the NeuralNetwork class.


class NeuralNetwork:
    def __init__(self, weights_initializer, bias_initializer):
        # list which will contain the loss after training
        self.loss = []
        self.data_layer = None   # the layer providing data
        self.loss_layer = None   # the layer calculating the loss and the prediction
        self.layers = []
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.label_tensor = None # the labels of the current iteration

    def append_fixed_layer(self, layer):
        """ Add a non-trainable layer to the network. """
        self.layers.append(layer)
    
    def append_trainable_layer(self, layer):
        """ Add a new layer with trainable parameters to the network. Initialize the parameters of 
        the network using the object's initializers for weights and bias.
        """
        layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def forward(self):
        """ Compute the forward pass through the network. """
        # fetch some training data
        input_tensor, self.label_tensor = self.data_layer.forward()
        # defer iterating through the network
        activation_tensor = self.__forward_input(input_tensor)
        # calculate the loss of the network using the final loss layer
        return self.loss_layer.forward(activation_tensor, self.label_tensor)

    def __forward_input(self, input_tensor):
        """ Compute the forward pass through the network, stopping before the 
            loss layer.
            param: input_tensor (np.ndarray): input to the network
            returns: activation of the last "regular" layer
        """
        activation_tensor = input_tensor
        # pass the input up the network
        for layer in self.layers:
            activation_tensor = layer.forward(activation_tensor)
        # return the activation of the last layer
        return activation_tensor

    def backward(self):
        """ Perform the backward pass during training. """
        error_tensor = self.loss_layer.backward(self.label_tensor)
        # pass back the error recursively
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def train(self, iterations):
        """ Train the network for a fixed number of steps.
            param: iterations (int): number of iterations for training 
        """
        for layer in self.layers:
            layer.phase = Phase.train  # Make sure phase is set to "train" for all layers
        for i in range(iterations):
            loss = self.forward()  # go up the network
            self.loss.append(loss)  # save the loss
            self.backward()  # and down again
            print('.', end='')


    def test(self, input_tensor):
        """ Apply the (trained) network to input data to generate a prediction. 
            param: input_tensor (nd.nparray): input (image or vector)
            returns (np.ndarray): prediction by the network
        """
        for layer in self.layers:
            layer.phase = Phase.test  # Make sure phase is set to "test" for all layers
        activation_tensor = self.__forward_input(input_tensor)
        return self.loss_layer.predict(activation_tensor)