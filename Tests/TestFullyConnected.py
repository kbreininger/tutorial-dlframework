import unittest
import numpy as np
import Helpers

class TestFullyConnected(unittest.TestCase):

    class TestInitializer:

        @staticmethod
        def initialize(weights_shape):
            return np.random.rand(*weights_shape)

    def setUp(self):
        self.batch_size = 9
        self.input_size = 4
        self.output_size = 3
        self.input_tensor = np.random.rand(self.batch_size, self.input_size)

        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_forward_size(self):
        layer = self.FullyConnected(self.input_size, self.output_size, 0)
        layer.initialize(TestFullyConnected.TestInitializer(),TestFullyConnected.TestInitializer)
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1], self.output_size)
        self.assertEqual(output_tensor.shape[0], self.batch_size)

    def test_backward_size(self):
        layer = self.FullyConnected(self.input_size, self.output_size, 0)
        layer.initialize(TestFullyConnected.TestInitializer(),TestFullyConnected.TestInitializer)
        output_tensor = layer.forward(self.input_tensor)
        error_tensor = layer.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], self.input_size)
        self.assertEqual(error_tensor.shape[0], self.batch_size)

    def test_update(self):
        layer = self.FullyConnected(self.input_size, self.output_size, 1)
        layer.initialize(TestFullyConnected.TestInitializer(),TestFullyConnected.TestInitializer)
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = np.zeros([ self.batch_size, self.output_size])
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        layers = list()
        layers.append(self.FullyConnected(self.input_size, self.categories, 0))
        layers[0].initialize(TestFullyConnected.TestInitializer(),TestFullyConnected.TestInitializer)
        layers.append(Helpers.SoftMax())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_weights(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        layers = list()
        layers.append(self.FullyConnected(self.input_size, self.categories, 0))
        layers[0].initialize(TestFullyConnected.TestInitializer(),TestFullyConnected.TestInitializer)
        layers.append(Helpers.SoftMax())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_bias(self):
        input_tensor = np.zeros((1, 100000))
        layer = self.FullyConnected(100000, 1, 0)
        layer.initialize(TestFullyConnected.TestInitializer(),TestFullyConnected.TestInitializer)
        result = layer.forward(input_tensor)
        self.assertGreater(np.sum(result), 0)

if __name__ == "__main__":
    pass