import unittest
import numpy as np
import Helpers
from scipy.ndimage.filters import gaussian_filter

class TestConv(unittest.TestCase):

    class TestInitializer:

        @staticmethod
        def initialize(weights):
            weights = np.zeros((1, 3, 3, 3))
            weights[0, 1, 1, 1] = 1
            return weights

    def setUp(self):
        self.batch_size = 2
        self.input_shape = (3, 10, 14)
        self.uneven_input_shape = (3, 11, 15)
        self.spatial_input_size = np.prod(self.input_shape[1:])
        self.kernel_shape = (3, 5, 8)
        self.num_kernels = 4
        self.hidden_channels = 3

        self.categories = 5
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_forward_size(self):
        conv = self.Conv( (1, 1), self.kernel_shape, self.num_kernels, 0.)
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_kernels, *self.input_shape[1:]))

    def test_forward_size_stride(self):
        conv = self.Conv((3, 2), self.kernel_shape, self.num_kernels, 0.)
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_kernels, 4, 7))

    def test_forward_size_stride_uneven_image(self):
        conv = self.Conv((3, 2), self.kernel_shape, self.num_kernels + 1, 0.)
        input_tensor = np.array(range(np.prod(self.uneven_input_shape) * (self.batch_size + 1)), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size + 1, *self.uneven_input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, ( self.batch_size+1, self.num_kernels+1, 4, 8))

    def test_forward(self):
        np.random.seed(1337)
        conv = self.Conv((1, 1), (1, 3, 3), 1, 0.)
        conv.weights = (1./15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.bias = np.array([0])
        conv.weights = np.expand_dims(conv.weights, 0)
        input_tensor = np.random.random((1, 1, 10, 14))
        expected_output = gaussian_filter(input_tensor[0, 0, :, :], 0.85, mode='constant', cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor))
        self.assertAlmostEqual(difference, 0., places=1)

    def test_forward_fully_connected_channels(self):
        np.random.seed(1337)
        conv = self.Conv((1, 1), (3, 3, 3), 1, 0.)
        conv.weights = (1. / 15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]], [[1, 2, 1], [2, 3, 2], [1, 2, 1]], [[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.bias = np.array([0])
        conv.weights = np.expand_dims(conv.weights, 0)
        tensor = np.random.random((1, 1, 10, 14))
        input_tensor = np.zeros((1, 3 , 10, 14))
        input_tensor[:,0] = tensor.copy()
        input_tensor[:,1] = tensor.copy()
        input_tensor[:,2] = tensor.copy()
        expected_output = 3 * gaussian_filter(input_tensor[0, 0, :, :], 0.85, mode='constant', cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor))
        self.assertLess(difference, 0.2)

    def test_backward_size(self):
        conv = self.Conv((1, 1), self.kernel_shape, self.num_kernels, 0.)
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape, (self.batch_size, *self.input_shape))

    def test_backward_size_stride(self):
        conv = self.Conv((3, 2), self.kernel_shape, self.num_kernels, 0.)
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape, (self.batch_size, *self.input_shape))
        
    def test_layout_preservation(self):
        conv = self.Conv((1, 1), (3, 3, 3), 1, 0.)
        conv.initialize(TestConv.TestInitializer(), self.Constant(0.0))
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertAlmostEqual(np.sum(np.abs(np.squeeze(output_tensor)-input_tensor[:,1,:,:])), 0.)

    def test_gradient(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(self.Conv((1, 1), (3, 3, 3), self.hidden_channels, 0.))
        layers.append(self.Flatten())
        layers.append(self.FullyConnected(35 * self.hidden_channels, self.categories, 0))
        layers.append(Helpers.SoftMax())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 5e-2)

    def test_gradient_weights(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(self.Conv((1, 1), (3, 3, 3), self.hidden_channels, 0.))
        layers.append(self.Flatten())
        layers.append(self.FullyConnected(35*self.hidden_channels, self.categories, 0))
        layers.append(Helpers.SoftMax())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)

        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_weights_strided(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(self.Conv((2, 2), (3, 3, 3), self.hidden_channels, 0.))
        layers.append(self.Flatten())
        layers.append(self.FullyConnected(12*self.hidden_channels, self.categories, 0))
        layers.append(Helpers.SoftMax())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)

        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_bias(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(self.Conv((1, 1), (3, 3, 3), self.hidden_channels, 0.))
        layers.append(self.Flatten())
        layers.append(self.FullyConnected(35 * self.hidden_channels, self.categories, 0))
        layers.append(Helpers.SoftMax())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, True)

        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_stride(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 14)))
        layers = list()
        layers.append(self.Conv( (1, 2), (3, 3, 3), 1, 0.))
        layers.append(self.Flatten())
        layers.append(self.FullyConnected(35, self.categories, 0))
        layers.append(Helpers.SoftMax())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)

        self.assertLessEqual(np.sum(difference), 1e-4)

    def test_update(self):
        input_tensor = np.abs(np.random.random((self.batch_size, *self.input_shape)))
        conv = self.Conv((3, 2), self.kernel_shape, self.num_kernels, 1.)
        conv.initialize(self.He(), self.Constant(0.1))
        for _ in range(10):
            output_tensor = conv.forward(input_tensor)
            error_tensor = np.zeros_like(output_tensor)
            error_tensor -= output_tensor
            conv.backward(error_tensor)
            new_output_tensor = conv.forward(input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))	

if __name__ == "__main__":
    pass